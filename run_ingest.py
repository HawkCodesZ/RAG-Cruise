import asyncio
from app.ingest import ingest_tickets

async def main():
    result = await ingest_tickets()
    print(result.model_dump_json(indent=2))

asyncio.run(main())