"""LiteLLM proxy hook: make Claude Code's request shape acceptable to Qwen's chat template.

The CLI sends the system prompt as several Anthropic content blocks (base prompt,
--append-system-prompt, cache-control segments). LiteLLM translates each block into its
own OpenAI `system` message, and Qwen3.5's chat template raises
`System message must be at the beginning.` on the second one.

Anthropic's own API concatenates multiple system blocks, so joining them with a blank line
reproduces the prompt the paper's proposer sees byte for byte -- nothing is dropped or
reordered. Purely a wire-format adapter; no experiment semantics live here.
"""
from litellm.integrations.custom_logger import CustomLogger


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type", "text") == "text"
        )
    return "" if content is None else str(content)


class MergeSystemMessages(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        # /v1/messages arrives in Anthropic shape: `system` is a list of blocks, and LiteLLM
        # turns each block into its own OpenAI system message. Collapse it here first.
        sys_field = data.get("system")
        if isinstance(sys_field, list):
            joined = "\n\n".join(
                b.get("text", "") for b in sys_field
                if isinstance(b, dict) and b.get("type", "text") == "text"
            )
            data["system"] = joined
            print("[proxy_hooks] collapsed %d system blocks (%d chars), call_type=%s"
                  % (len(sys_field), len(joined), call_type), flush=True)
        messages = data.get("messages")
        if not messages:
            return data
        # The CLI also puts `system` entries inside `messages` (its agent listing, reminders).
        # Moved to the front as a system message, as this hook used to do, LiteLLM then inserts
        # the `system` field ahead of it and the provider sees two system messages, which the
        # stock Qwen3.5 template rejects. The local server ran config/qwen35_chat_template.jinja,
        # which concatenates every system message into the leading block; DeepInfra's does not
        # (2026-10-12 08:03). Do the same concatenation here, so the prompt renders as it did.
        extra, rest = [], []
        for m in messages:
            if m.get("role") == "system":
                t = _text(m.get("content"))
                if t:
                    extra.append(t)
            else:
                rest.append(m)
        if extra:
            head = [data["system"]] if data.get("system") else []
            data["system"] = "\n\n".join(head + extra)
            data["messages"] = rest
        return data


proxy_handler_instance = MergeSystemMessages()
