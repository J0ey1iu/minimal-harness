# Vision

## The Idea

I want a minimal structure that can clearly define an **agent** and a bunch of them to form a **team** — multiple agents in one shared space.

The concepts should be clean. **Agent**, **Tool**, **Memory** as fundamentals. No sub-agents, no fancy derivative stuff. All primitive and easy to understand.

All agents are equal. No one is another's subordinate. There will be handoffs in a team, but all agents are peers.

## Phase 1: What Is an Agent

First, I need to figure out what an agent actually is. I want to focus on building a reasonable agent concept layer without worrying about how multi-agent systems operate.

Every agent is an **event emitter**. All UIs or applications that visualize an agent at work are consumers of the events it generates. It's their job to process events and decide how to visualize them. That's not the vital part of this project — but I will build some simple implementations to verify the project works. Those applications are just proof of concept, not the product itself.

## Phase 2: Agent Service

At some point, I'll need to think about multi-agent collaboration. Every agent needs a way of receiving messages and processing them. I'll also need to think about using this in production — deploying it. So an **agent service** is bound to emerge. It is the entry point of each agent.

## Phase 3: The Full System

Eventually, this project will become a **system** of agents working together.

They will have a way to sense the existence of each other, get to know each other, ask for each other's help, listen and handoff to one another — a registry of agents, communication tunnels, and all those collaboration patterns.

That's far away from where we are now. But that's the direction.
