import { describe, expect, test } from "bun:test";

import { recentConversation } from "../src/conversation.ts";
import { asExtensionContext } from "./helpers/extension-context.ts";

const LOW_SURROGATE_START = /^[\uDC00-\uDFFF]/u;

const assistant = (content: unknown[]) => ({
  message: { content, role: "assistant" },
  type: "message",
});

const branchContext = (entries: unknown[]) =>
  asExtensionContext({ sessionManager: { getBranch: () => entries } });

const thinking = (
  text: string,
  extra: { redacted?: boolean; thinkingSignature?: string } = {}
) => ({
  thinking: text,
  type: "thinking",
  ...extra,
});

const withReasoning = (id: string, text: string) => ({
  id,
  ...assistant([thinking(text)]),
});

const redactedDigest = (reasoning: string) =>
  recentConversation(
    branchContext([assistant([thinking(reasoning)])]),
    15_000,
    2000,
    50 * 1024,
    {},
    true
  );

describe("Executor reasoning in the conversation digest", () => {
  test("labels reasoning before the visible reply and tool calls", () => {
    const result = recentConversation(
      branchContext([
        { message: { content: "fix it", role: "user" }, type: "message" },
        assistant([
          thinking("The test fails because the cache is stale.", {
            thinkingSignature: "opaque-signature-payload",
          }),
          { text: "Clearing the cache.", type: "text" },
          {
            arguments: { command: "rm -rf .cache" },
            name: "bash",
            type: "toolCall",
          },
        ]),
      ])
    );
    expect(result).toContain(
      'Executor: [Reasoning] (not shown to the user):\nThe test fails because the cache is stale.\n[End of reasoning]\nClearing the cache.\n[Tool Call: bash({"command":"rm -rf .cache"})]'
    );
    expect(result).not.toContain("opaque-signature-payload");
  });

  test("skips redacted, opaque, and empty thinking blocks", () => {
    const result = recentConversation(
      branchContext([
        assistant([
          thinking("[Reasoning redacted]", {
            redacted: true,
            thinkingSignature: "encrypted-payload",
          }),
          thinking("   ", { thinkingSignature: "encrypted-reasoning-item" }),
          { thinkingSignature: "signature-only", type: "thinking" },
          { text: "Done.", type: "text" },
        ]),
      ])
    );
    expect(result).toBe("Executor: Done.");
  });

  test("keeps a reasoning-only turn, such as one cut off mid-thought", () => {
    const result = recentConversation(
      branchContext([assistant([thinking("Still looping on the parser.")])])
    );
    expect(result).toBe(
      "Executor: [Reasoning] (not shown to the user):\nStill looping on the parser.\n[End of reasoning]"
    );
  });

  test("keeps the end of long reasoning and states how much was omitted", () => {
    const reasoning = `${"a".repeat(10_000)}${"b".repeat(3750)}`;
    const result = recentConversation(
      branchContext([assistant([thinking(reasoning)])]),
      15_000
    );
    expect(result).toContain(
      `[... 10000 earlier reasoning characters omitted ...]\n${"b".repeat(3750)}\n[End of reasoning]`
    );
    expect(result).not.toContain("aaa");
    const unbounded = recentConversation(
      branchContext([assistant([thinking(reasoning)])]),
      Number.MAX_SAFE_INTEGER
    );
    expect(unbounded).toContain(
      "[... 9750 earlier reasoning characters omitted ...]"
    );
  });

  test("caps the combined reasoning of one turn", () => {
    const result = recentConversation(
      branchContext([
        assistant([thinking("x".repeat(3000)), thinking("y".repeat(3000))]),
      ]),
      15_000
    );
    expect(result).toContain(
      "[... 2251 earlier reasoning characters omitted ...]"
    );
    expect(result).toContain(`${"x".repeat(749)}\n${"y".repeat(3000)}`);
  });

  test("never starts the kept reasoning inside a surrogate pair", () => {
    const result = recentConversation(
      branchContext([assistant([thinking("😀".repeat(3000))])]),
      15_000
    );
    const kept = result.split("omitted ...]\n")[1] ?? "";
    expect(LOW_SURROGATE_START.test(kept)).toBe(false);
    expect(kept.startsWith("😀")).toBe(true);
  });

  test("redacts reasoning before capping it", () => {
    const secret = "AKIAABCDEFGHIJKLMNOP";
    const straddling = redactedDigest(
      `${"a".repeat(4000)} token=${secret} ${"b".repeat(3740)}`
    );
    expect(straddling).toContain("[... 4009 earlier reasoning characters");
    expect(straddling).not.toContain("HIJKLMNOP");
    const whole = redactedDigest(`Retry with token=${secret} next.`);
    expect(whole).not.toContain(secret);
    expect(whole).toContain("Retry with [REDACTED SECRET] next.");
  });

  test("keeps several reasoning turns within a small digest budget", () => {
    const turns = Array.from({ length: 4 }, (_, index) => [
      assistant([
        thinking(`${index}`.repeat(5000)),
        { arguments: { index }, name: "bash", type: "toolCall" },
      ]),
      {
        message: {
          content: `result ${index}`,
          role: "toolResult",
          toolName: "bash",
        },
        type: "message",
      },
    ]).flat();
    const result = recentConversation(branchContext(turns), 4000);
    expect(result.length).toBeLessThanOrEqual(4000);
    expect(result).toContain(
      "[... 4000 earlier reasoning characters omitted ...]"
    );
    expect(result).toContain("result 3");
    expect(result).toContain("result 2");
    expect(result.match(/\[Reasoning\]/gu)?.length).toBeGreaterThan(1);
  });

  test("forwards no reasoning from compacted-out or reset entries", () => {
    const result = recentConversation(
      branchContext([
        withReasoning("cleared", "reasoning before the reset"),
        { id: "reset", type: "reset_boundary" },
        withReasoning("old", "compacted private reasoning"),
        withReasoning("kept", "retained reasoning"),
        {
          firstKeptEntryId: "kept",
          id: "compaction",
          parentId: null,
          summary: "safe summary",
          timestamp: "2026-01-01T00:00:00Z",
          tokensBefore: 1000,
          type: "compaction",
        },
        withReasoning("new", "current reasoning"),
      ])
    );
    expect(result).not.toContain("reasoning before the reset");
    expect(result).not.toContain("compacted private reasoning");
    expect(result).toContain("retained reasoning");
    expect(result).toContain("current reasoning");
  });
});
