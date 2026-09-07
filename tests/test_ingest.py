import pandas as pd
import pytest

from app.models import IngestResponse


@pytest.mark.asyncio
async def test_reingest_reports_updates_not_additions(monkeypatch, mock_embed, mock_collection):
    """Re-ingesting tickets that already exist in the store should report
    them as updated, not added — this is what makes the no-duplication
    requirement demonstrable rather than just asserted."""
    fake_df = pd.DataFrame([
        {
            "Number": "INC0000001", "Category": "Hardware", "Priority": "1 - Critical",
            "Location": "SU - Sun Princess", "Short description": "Test ticket",
            "Description": "A test description.", "Resolution notes": "Fixed it.",
            "Assignment group": "IT", "Incident state": "Closed", "Resolution code": "Fixed",
        },
    ])
    monkeypatch.setattr(pd, "read_excel", lambda path: fake_df)
    monkeypatch.setattr("app.ingest.get_collection", lambda: mock_collection)
    monkeypatch.setattr("app.vectorstore.get_collection", lambda: mock_collection)  # add this
    monkeypatch.setattr("app.ingest.collection_count", lambda: 1)
    monkeypatch.setattr("app.ingest.export_snapshot", lambda: None)

    # Simulate this ticket ID already existing in the store.
    mock_collection.get.return_value = {"ids": ["INC0000001"]}

    from app.ingest import ingest_tickets
    result = await ingest_tickets(xlsx_path="fake/path.xlsx")

    assert isinstance(result, IngestResponse)
    assert result.chunks_added == 0
    assert result.chunks_updated == 1
    mock_collection.upsert.assert_called_once()
    call_kwargs = mock_collection.upsert.call_args.kwargs
    assert call_kwargs["ids"] == ["INC0000001"]