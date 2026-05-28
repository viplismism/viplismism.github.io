---
title: "mcp"
source: obsidian
---

MCP, or Model Context Protocol, is a protocol for connecting models and agents to external tools, data sources, and services in a structured way.

The point is to avoid every tool integration becoming a custom one-off adapter.

An MCP server can expose capabilities such as:

```text
read files
query databases
call APIs
search documents
inspect application state
execute domain tools
```

The model or agent does not directly know every API detail. It receives a structured interface describing available tools and how to call them.

Agents become useful when they can act on real systems. MCP-style integration makes tool access more standard, inspectable, and reusable.

The risk is security and permissions. Tool access must be scoped carefully. A model with broad uncontrolled tool access is not an agent; it is an incident waiting to happen.

Related: [agents](/glossary/agents), [context engineering](/glossary/context-engineering), [deployment](/glossary/deployment).
