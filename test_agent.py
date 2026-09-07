import asyncio
from app.agent import run_agent

async def main():
    # Should call search_tickets, cite ticket(s)
    a1, s1, t1 = await run_agent("Have we had any issues with stateroom TVs going to a black screen?")
    print("Q1 tool:", t1, "| sources:", [s.chunk_id for s in s1])
    print(a1, "\n")

    # Should answer directly, no tool call
    a2, s2, t2 = await run_agent("What's the capital of France?")
    print("Q2 tool:", t2, "| sources:", s2)
    print(a2, "\n")

    # Should call a tool, get nothing, honestly say it doesn't know
    a3, s3, t3 = await run_agent("What is our company's refund policy for cruise cancellations?")
    print("Q3 tool:", t3, "| sources:", s3)
    print(a3)

asyncio.run(main())
