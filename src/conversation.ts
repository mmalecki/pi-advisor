import type { ExtensionContext } from "@earendil-works/pi-coding-agent";

import {
  advisorRedactSecretsRef,
  advisorToolPoliciesRef,
  advisorToolResultMaxBytesRef,
  advisorToolResultMaxLinesRef,
} from "./config/state.ts";
import type { AdvisorToolPolicies } from "./config/types.ts";
import { contentParts, isRecordOf, isString } from "./content-utils.ts";
import type { RecordValue } from "./content-utils.ts";
import { imageMarker } from "./images.ts";
import { redactSecrets } from "./redaction.ts";
import { sessionContextEntries } from "./session-context.ts";
import { capToolResult } from "./tool-result-cap.ts";

const textFromPart = <Part>(part: Part): string => {
  if (isString(part)) {
    return part;
  }
  if (!isRecordOf(part) || part.type !== "text") {
    return "";
  }
  return isString(part.text) ? part.text : "";
};

export const textFrom = <Content>(content: Content): string =>
  contentParts(content).map(textFromPart).join("\n").trim();

const REASONING_MAX_CHARS = 4000;

export const reasoningBudget = (maxChars: number) =>
  Math.min(REASONING_MAX_CHARS, Math.floor(maxChars / 4));

const reasoningFrom = (content: RecordValue["content"]) =>
  contentParts(content)
    .map((part) =>
      isRecordOf(part) &&
      part.type === "thinking" &&
      part.redacted !== true &&
      isString(part.thinking)
        ? part.thinking.trim()
        : ""
    )
    .filter(Boolean)
    .join("\n");

const tailWithinChars = (value: string, maxChars: number) => {
  const start = Math.max(0, value.length - maxChars);
  const code = value.codePointAt(start) ?? 0;
  return value.slice(code >= 0xdc_00 && code <= 0xdf_ff ? start + 1 : start);
};

const reasoningPart = (
  reasoning: string,
  maxChars: number,
  redact: boolean
) => {
  const disclosed = redact ? redactSecrets(reasoning) : reasoning;
  const kept = tailWithinChars(disclosed, maxChars);
  const omitted = disclosed.length - kept.length;
  return [
    "[Reasoning] (not shown to the user):",
    omitted > 0
      ? `[... ${omitted} earlier reasoning characters omitted ...]`
      : "",
    kept,
    "[End of reasoning]",
  ]
    .filter(Boolean)
    .join("\n");
};

const imageNotes = (content: RecordValue["content"], nonce: string) =>
  contentParts(content)
    .map((part) => (isRecordOf(part) ? imageMarker(part, nonce) : undefined))
    .filter(isString);

const assistantEntry = (
  message: RecordValue,
  policies: AdvisorToolPolicies,
  redact: boolean,
  reasoningMaxChars: number
): string | undefined => {
  const parts: string[] = [];
  const reasoning = reasoningFrom(message.content);
  if (reasoning) {
    parts.push(reasoningPart(reasoning, reasoningMaxChars, redact));
  }
  const text = textFrom(message.content);
  if (text) {
    parts.push(redact ? redactSecrets(text) : text);
  }
  for (const part of contentParts(message.content)) {
    if (!isRecordOf(part) || part.type !== "toolCall") {
      continue;
    }
    const toolName = isString(part.name) ? part.name : "unknown";
    const policy = policies[toolName] ?? "full";
    if (policy === "exclude") {
      parts.push(`[Tool Call: ${toolName}] (excluded by Advisor tool policy)`);
      continue;
    }
    if (policy === "summary") {
      parts.push(
        `[Tool Call: ${toolName}] (arguments omitted by Advisor tool policy: summary)`
      );
      continue;
    }
    const argumentsText = JSON.stringify(part.arguments) ?? "undefined";
    parts.push(
      `[Tool Call: ${toolName}(${redact ? redactSecrets(argumentsText) : argumentsText})]`
    );
  }
  return parts.length > 0 ? `Executor: ${parts.join("\n")}` : undefined;
};

const toolResultEntry = (
  message: RecordValue,
  toolResultMaxLines: number,
  toolResultMaxBytes: number,
  policies: AdvisorToolPolicies,
  redact: boolean,
  describeImages: boolean,
  imageNonce: string
): string => {
  const status = message.isError ? "error" : "success";
  const toolName = isString(message.toolName) ? message.toolName : "unknown";
  const policy = policies[toolName] ?? "full";
  const notes = describeImages ? imageNotes(message.content, imageNonce) : [];
  const source = [textFrom(message.content), ...notes]
    .filter(Boolean)
    .join("\n");
  if (policy === "exclude") {
    return `[Tool Result for ${toolName}] (excluded by Advisor tool policy)`;
  }
  if (policy === "summary") {
    const capped = capToolResult(
      source,
      toolResultMaxLines,
      toolResultMaxBytes
    );
    return `[Tool Result for ${toolName}] (output omitted by Advisor tool policy: summary; status: ${status}; ${capped.totalLines} lines, ${capped.totalBytes} bytes; source output was${capped.truncated ? "" : " not"} truncated)`;
  }
  const disclosed = redact ? redactSecrets(source) : source;
  const capped = capToolResult(
    disclosed,
    toolResultMaxLines,
    toolResultMaxBytes
  );
  return `[Tool Result for ${toolName}] (${message.isError ? "Error " : ""}output):\n${capped.content}`;
};

export const conversationEntry = <Entry>(
  entry: Entry,
  toolResultMaxLines: number,
  toolResultMaxBytes: number,
  policies: AdvisorToolPolicies,
  redact: boolean,
  describeImages = false,
  imageNonce = "",
  reasoningMaxChars = REASONING_MAX_CHARS
): string | undefined => {
  if (!isRecordOf(entry)) {
    return;
  }
  if (entry.type === "compaction" && isString(entry.summary)) {
    return `[System Compaction Summary]: ${redact ? redactSecrets(entry.summary) : entry.summary}`;
  }
  if (entry.type !== "message" || !isRecordOf(entry.message)) {
    return;
  }
  const { message } = entry;
  if (message.role === "user") {
    const notes = describeImages ? imageNotes(message.content, imageNonce) : [];
    const text = [textFrom(message.content), ...notes]
      .filter(Boolean)
      .join("\n");
    return text ? `User: ${redact ? redactSecrets(text) : text}` : undefined;
  }
  if (message.role === "assistant") {
    return assistantEntry(message, policies, redact, reasoningMaxChars);
  }
  if (message.role === "toolResult" || message.role === "tool") {
    return toolResultEntry(
      message,
      toolResultMaxLines,
      toolResultMaxBytes,
      policies,
      redact,
      describeImages,
      imageNonce
    );
  }
};

const omissionMarker = (omitted: number) =>
  `[Older context omitted: ${omitted} complete entr${omitted === 1 ? "y" : "ies"}]`;

const selectRecentEntries = (entries: string[], maxChars: number): string => {
  const separator = "\n\n";
  const joined = entries.join(separator);
  if (joined.length <= maxChars || maxChars === Number.MAX_SAFE_INTEGER) {
    return joined;
  }
  const newestTruncated = "[Newest entry truncated]";
  if (entries.length === 1) {
    const prefix = `${newestTruncated}${separator}`;
    return `${prefix}${entries[0].slice(0, Math.max(0, maxChars - prefix.length))}`.slice(
      0,
      maxChars
    );
  }
  const selected: string[] = [];
  let selectedLength = 0;
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index];
    const candidateCount = selected.length + 1;
    const omitted = entries.length - candidateCount;
    const candidateLength =
      selectedLength +
      entry.length +
      (selected.length > 0 ? separator.length : 0);
    if (
      omissionMarker(omitted).length + separator.length + candidateLength >
      maxChars
    ) {
      break;
    }
    selected.unshift(entry);
    selectedLength = candidateLength;
  }
  const omitted = entries.length - Math.max(1, selected.length);
  const marker = omissionMarker(omitted);
  if (selected.length > 0) {
    return `${marker}${separator}${selected.join(separator)}`;
  }
  const prefix = `${marker}${separator}${newestTruncated}${separator}`;
  return `${prefix}${entries.at(-1)?.slice(0, Math.max(0, maxChars - prefix.length)) ?? ""}`.slice(
    0,
    maxChars
  );
};

export const recentConversation = (
  ctx: ExtensionContext,
  maxChars = 15_000,
  toolResultMaxLines = advisorToolResultMaxLinesRef,
  toolResultMaxBytes = advisorToolResultMaxBytesRef,
  policies = advisorToolPoliciesRef,
  redact = advisorRedactSecretsRef,
  describeImages = false,
  imageNonce = ""
): string => {
  if (maxChars === 0) {
    return "";
  }
  const entries = sessionContextEntries(ctx)
    .map((entry) =>
      conversationEntry(
        entry,
        toolResultMaxLines,
        toolResultMaxBytes,
        policies,
        redact,
        describeImages,
        imageNonce,
        reasoningBudget(maxChars)
      )
    )
    .filter((entry): entry is string => entry !== undefined);
  return selectRecentEntries(entries, maxChars);
};
