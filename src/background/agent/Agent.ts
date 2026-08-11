import {
  TextGenerationPipeline,
  TextStreamer,
  pipeline,
} from "@huggingface/transformers";

import { MODELS, TEXT_GENERATION_ID } from "../../shared/constants.ts";
import {
  AgentMetrics,
  ChatMessage,
  ChatMessageAssistant,
} from "../../shared/types.ts";
import { extractToolCalls } from "./extractToolCalls.ts";
import { ToolCallPayload } from "./types.ts";
import {
  WebMCPTool,
  executeWebMCPTool,
  webMCPToolToChatTemplateTool,
} from "./webMcp.tsx";

type Message = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  [key: string]: any;
};

type GenerationMetrics = AgentMetrics;
export type AgentRunMetrics = AgentMetrics;

let pipe: TextGenerationPipeline | null = null;
const SYSTEM_PROMPT =
  "You are a helpful assistant with access to external tools declared in this conversation. " +
  "Never claim you do not have tools when tool declarations are present. " +
  "When asked what tools you have, list the declared tool names exactly. " +
  "You may call multiple tools in parallel or sequentially over multiple steps to complete a task. " +
  "If you decide to use a tool, briefly explain what you are doing before calling it. " +
  "After receiving tool results, evaluate whether additional tool calls are needed or if you have enough information to answer. " +
  "When the user asks to summarize, analyze, or ask questions about 'the page', 'the tab', or the current website: first call get_open_tabs to find the active tab, and then call ask_website using that tab's ID to retrieve the page content before generating your final response.";
const createInitialMessages = (): Array<Message> => [
  {
    role: "system",
    content: SYSTEM_PROMPT,
  },
];
const END_OF_TEXT_TOKEN_REGEX = /<\|end_of_text\|>|<turn\|>|<end_of_turn>/g;
const sanitizeModelText = (text: string) =>
  text.replace(END_OF_TEXT_TOKEN_REGEX, "").trim();

const getTextGenerationPipeline = async (
  onDownloadProgress: (id: string, percentage: number) => void = () => {}
): Promise<TextGenerationPipeline> => {
  if (pipe) return pipe;

  try {
    const m = MODELS[TEXT_GENERATION_ID];
    pipe = (await pipeline("text-generation", m.modelId, {
      dtype: m.dtype,
      device: "webgpu",
      progress_callback: (i) => {
        if (i.status === "progress_total") {
          onDownloadProgress(m.modelId, i.progress);
        }
      },
    })) as TextGenerationPipeline;

    return pipe;
  } catch (error) {
    console.error("Failed to initialize text generation pipeline:", error);
    throw error;
  }
};

class Agent {
  private messages: Array<Message> = createInitialMessages();
  private _chatMessages: Array<ChatMessage> = [];
  private chatMessagesListener: Array<
    (chatMessages: Array<ChatMessage>) => void
  > = [];
  private tools: Array<WebMCPTool> = [];

  constructor() {}

  get chatMessages() {
    return this._chatMessages;
  }

  set chatMessages(chatMessages: Array<ChatMessage>) {
    this._chatMessages = chatMessages;
    this.chatMessagesListener.forEach((listener) => listener(chatMessages));
  }

  public onChatMessageUpdate(callback: (messages: Array<ChatMessage>) => void) {
    this.chatMessagesListener.push(callback);
  }

  public setTool = (tool: WebMCPTool) => {
    this.tools = [...this.tools, tool];
  };

  public getTextGenerationPipeline = getTextGenerationPipeline;

  public generateText = async (
    prompt: string,
    role: "user" | "tool" = "user",
    onResponseUpdate: (response: string) => void = () => {},
    options: { appendPromptMessage?: boolean } = {}
  ): Promise<{ text: string; metrics: GenerationMetrics }> => {
    const start = performance.now();
    let firstTokenAt: number | null = null;

    if (!this.messages.some(({ role }) => role === "system")) {
      this.messages = [...createInitialMessages(), ...this.messages];
    }

    if (options.appendPromptMessage ?? true) {
      this.messages = [...this.messages, { role, content: prompt }];
    }
    const pipe = await this.getTextGenerationPipeline();
    const conversation = [...this.messages];
    let response = "";

    // Add placeholder assistant message for streaming UI updates
    this.messages.push({ role: "assistant", content: "" });

    const streamer = new TextStreamer(pipe.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: false,
      callback_function: (token: string) => {
        if (firstTokenAt === null) {
          firstTokenAt = performance.now();
        }
        response = response + token;
        this.messages = this.messages.map((message, index, all) => ({
          ...message,
          content: index === all.length - 1 ? response : message.content,
        }));
        onResponseUpdate(sanitizeModelText(response));
      },
    });

    const input = pipe.tokenizer.apply_chat_template(conversation, {
      tools: this.tools.map(webMCPToolToChatTemplateTool),
      add_generation_prompt: true,
      return_dict: true,
    }) as any;

    const output: any = await pipe(conversation, {
      tools: this.tools.map(webMCPToolToChatTemplateTool),
      add_generation_prompt: true,
      max_new_tokens: 1024,
      do_sample: false,
      streamer,
    });

    const promptLength = Number(input.input_ids.dims.at(-1) ?? 0);
    const finalGeneratedText = output?.[0]?.generated_text;

    if (Array.isArray(finalGeneratedText) && response.trim().length === 0) {
      const lastMessage = finalGeneratedText[finalGeneratedText.length - 1];
      if (typeof lastMessage === "string") {
        response = lastMessage;
      } else {
        const content =
          typeof lastMessage?.content === "string" ? lastMessage.content : "";
        const toolCalls = Array.isArray(lastMessage?.tool_calls)
          ? lastMessage.tool_calls
          : [];

        if (toolCalls.length > 0) {
          const renderedToolCalls = toolCalls
            .map((toolCall: any) => {
              const functionName = toolCall?.function?.name;
              const functionArguments = toolCall?.function?.arguments ?? {};
              if (typeof functionName !== "string" || !functionName.trim()) {
                return "";
              }

              const serializedArguments =
                typeof functionArguments === "string"
                  ? functionArguments
                  : JSON.stringify(functionArguments);

              return `<|tool_call>call:${functionName}${serializedArguments}<tool_call|>`;
            })
            .filter(Boolean)
            .join("");

          if (renderedToolCalls) response = renderedToolCalls;
          else if (content.length > 0) response = content;
        } else if (content.length > 0) {
          response = content;
        }
      }
    }

    const generatedIds: any = pipe.tokenizer(response, {
      add_special_tokens: false,
    }).input_ids;
    const generatedTokens = Array.isArray(generatedIds?.[0])
      ? generatedIds[0].length
      : Array.isArray(generatedIds)
        ? generatedIds.length
        : 0;

    response = sanitizeModelText(response);

    this.messages = this.messages.map((message, index, all) => ({
      ...message,
      content: index === all.length - 1 ? response : message.content,
    }));

    const end = performance.now();
    const prefillMs = Math.max(0, (firstTokenAt ?? end) - start);
    const totalMs = Math.max(0, end - start);
    const decodeMs = Math.max(0, totalMs - prefillMs);

    const metrics: GenerationMetrics = {
      generatedTokens,
      prefillTokens: promptLength,
      prefillMs,
      prefillTokensPerSecond:
        prefillMs > 0 ? promptLength / (prefillMs / 1000) : 0,
      decodeMs,
      totalMs,
      tokensPerSecond: decodeMs > 0 ? generatedTokens / (decodeMs / 1000) : 0,
      msPerToken: generatedTokens > 0 ? decodeMs / generatedTokens : 0,
    };

    return { text: response, metrics };
  };

  public runAgent = async (prompt: string): Promise<AgentRunMetrics> => {
    const MAX_AGENT_STEPS = 5;
    let currentStep = 0;
    let roleForGeneration: "user" | "tool" = "user";
    let appendPromptMessage = true;
    let currentPrompt: string | null = prompt;
    const start = performance.now();
    let generatedTokens = 0;
    let prefillTokens = 0;
    let prefillMs = 0;
    let decodeMs = 0;

    this.chatMessages = [
      ...this.chatMessages,
      { role: "user", content: prompt },
    ];
    const prevChatMessages = this.chatMessages;
    const assistantMessage: ChatMessageAssistant = {
      role: "assistant",
      content: "",
      tools: [],
      metrics: {
        generatedTokens: 0,
        prefillTokens: 0,
        prefillMs: 0,
        prefillTokensPerSecond: 0,
        decodeMs: 0,
        totalMs: 0,
        tokensPerSecond: 0,
        msPerToken: 0,
      },
    };

    this.chatMessages = [...prevChatMessages, assistantMessage];

    let messageInThisAgentRun = "";
    const updateAssistantMessage = (response: string) => {
      const { toolCalls, message } = extractToolCalls(response);

      toolCalls.forEach((tool) => {
        if (!assistantMessage.tools.some(({ id }) => tool.id === id)) {
          assistantMessage.tools = [
            ...assistantMessage.tools,
            {
              name: tool.name,
              functionSignature: `${tool.name}(${JSON.stringify(
                tool.arguments
              )})`,
              id: tool.id,
              result: "",
            },
          ];
        }
      });

      assistantMessage.content = messageInThisAgentRun
        ? `${messageInThisAgentRun}${message ? `\n\n${message}` : ""}`
        : message;

      this.chatMessages = [...prevChatMessages, assistantMessage];
    };

    while (currentPrompt !== null && currentStep < MAX_AGENT_STEPS) {
      currentStep += 1;
      const generation = await this.generateText(
        currentPrompt,
        roleForGeneration,
        updateAssistantMessage,
        { appendPromptMessage }
      );

      const finalResponse = generation.text;
      generatedTokens += generation.metrics.generatedTokens;
      prefillTokens += generation.metrics.prefillTokens;
      prefillMs += generation.metrics.prefillMs;
      decodeMs += generation.metrics.decodeMs;
      const elapsedMs = Math.max(0, performance.now() - start);
      assistantMessage.metrics = {
        generatedTokens,
        prefillTokens,
        prefillMs,
        prefillTokensPerSecond:
          prefillMs > 0 ? prefillTokens / (prefillMs / 1000) : 0,
        decodeMs,
        totalMs: elapsedMs,
        tokensPerSecond: decodeMs > 0 ? generatedTokens / (decodeMs / 1000) : 0,
        msPerToken: generatedTokens > 0 ? decodeMs / generatedTokens : 0,
      };

      const { toolCalls, message } = extractToolCalls(finalResponse);
      if (message) {
        messageInThisAgentRun = messageInThisAgentRun
          ? `${messageInThisAgentRun}\n\n${message}`
          : message;
      }

      if (toolCalls.length === 0) {
        currentPrompt = null;
      } else {
        const toolResponses = await Promise.all(
          toolCalls.map(this.executeToolCall)
        );

        // Update the assistant message in message history with structured tool calls and empty content
        // (Gemma chat template expects empty content so it does not prematurely close the turn)
        for (let i = this.messages.length - 1; i >= 0; i -= 1) {
          if (this.messages[i].role === "assistant") {
            this.messages[i] = {
              ...this.messages[i],
              content: "",
              tool_calls: toolCalls.map((call) => ({
                id: call.id,
                type: "function",
                function: {
                  name: call.name,
                  arguments: call.arguments,
                },
              })),
            };
            break;
          }
        }

        this.messages = [
          ...this.messages,
          ...toolResponses.map(({ id, name, result }) => ({
            role: "tool" as const,
            tool_call_id: id,
            name,
            content: typeof result === "string" ? result : JSON.stringify(result),
          })),
        ];

        assistantMessage.tools = assistantMessage.tools.map((tool) => ({
          ...tool,
          result:
            toolResponses.find(({ id }) => id === tool.id)?.result ||
            tool.result,
        }));

        this.chatMessages = [...prevChatMessages, assistantMessage];

        // Continue the conversation directly from the tool responses without injecting a fake user prompt
        currentPrompt = "";
        roleForGeneration = "tool";
        appendPromptMessage = false;
      }
    }
    const totalMs = Math.max(0, performance.now() - start);
    assistantMessage.metrics = {
      generatedTokens,
      prefillTokens,
      prefillMs,
      prefillTokensPerSecond:
        prefillMs > 0 ? prefillTokens / (prefillMs / 1000) : 0,
      decodeMs,
      totalMs,
      tokensPerSecond: decodeMs > 0 ? generatedTokens / (decodeMs / 1000) : 0,
      msPerToken: generatedTokens > 0 ? decodeMs / generatedTokens : 0,
    };
    this.chatMessages = [...prevChatMessages, assistantMessage];

    // Compact conversation history across turns:
    // Retain system message, user messages, and completed assistant text responses.
    // Prunes intermediate raw tool calls and bulky tool response payloads from previous turns
    // so total context token length stays safely within WebGPU sliding-window limits.
    const compactedMessages: Array<Message> = [];
    const systemMsg = this.messages.find((m) => m.role === "system");
    if (systemMsg) {
      compactedMessages.push(systemMsg);
    }

    for (const msg of this.messages) {
      if (msg.role === "user") {
        compactedMessages.push(msg);
      } else if (
        msg.role === "assistant" &&
        typeof msg.content === "string" &&
        msg.content.trim().length > 0
      ) {
        compactedMessages.push({
          role: "assistant",
          content: msg.content.trim(),
        });
      }
    }

    const maxHistoryTurns = 8;
    if (compactedMessages.length > maxHistoryTurns + 1) {
      const system = compactedMessages[0];
      const recent = compactedMessages.slice(-maxHistoryTurns);
      this.messages = [system, ...recent];
    } else {
      this.messages = compactedMessages;
    }

    return {
      generatedTokens,
      prefillTokens,
      prefillMs,
      prefillTokensPerSecond:
        prefillMs > 0 ? prefillTokens / (prefillMs / 1000) : 0,
      decodeMs,
      totalMs,
      tokensPerSecond: decodeMs > 0 ? generatedTokens / (decodeMs / 1000) : 0,
      msPerToken: generatedTokens > 0 ? decodeMs / generatedTokens : 0,
    };
  };

  private executeToolCall = async (
    toolCall: ToolCallPayload
  ): Promise<{ id: string; name: string; result: string }> => {
    const toolToUse = this.tools.find((t) => t.name === toolCall.name);
    if (!toolToUse)
      throw new Error(`Tool '${toolCall.name}' not found or is disabled.`);

    return {
      id: toolCall.id,
      name: toolCall.name,
      result: await executeWebMCPTool(toolToUse, toolCall.arguments),
    };
  };

  public clear() {
    this.messages = createInitialMessages();
    this.chatMessages = [];
  }
}

export default Agent;
