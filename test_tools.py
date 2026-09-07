import asyncio
from app.tools import search_tickets_impl, lookup_ticket_impl

async def main():
    r1 = await search_tickets_impl("TV screen goes black")
    print("search_tickets:", r1["found"], len(r1.get("matches", [])))

    r2 = await search_tickets_impl("what is the capital of France")
    print("search_tickets (should be empty):", r2)

    r3 = await lookup_ticket_impl(number="INC0200408")
    print("lookup_ticket by number:", r3["found"])

    r4 = await lookup_ticket_impl(category="Security", limit=3)
    print("lookup_ticket by category:", r4["found"], len(r4.get("results", [])))

asyncio.run(main())