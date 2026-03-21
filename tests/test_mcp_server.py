"""Tests for MCP server tool logic."""

from unittest.mock import MagicMock

import pytest

from bookmark_app.mcp_server import (
    AppContext,
    _get_bookmark_stats_logic,
    _list_bookmarks_logic,
    _search_bookmarks_logic,
)


SAMPLE_BOOKMARKS = [
    {
        "folder": "/Tools/Dev",
        "name": "GitHub",
        "url": "https://github.com",
        "description": "Code hosting and version control platform.",
    },
    {
        "folder": "/Learning/ML",
        "name": "fast.ai",
        "url": "https://fast.ai",
        "description": "Practical deep learning courses and library.",
    },
    {
        "folder": "/Tools/Dev",
        "name": "Stack Overflow",
        "url": "https://stackoverflow.com",
        "description": "Q&A site for programmers.",
    },
]


@pytest.fixture
def app_ctx():
    """Create an AppContext with sample data (no vector store)."""
    return AppContext(
        bookmarks=list(SAMPLE_BOOKMARKS),
        vector_store=None,
        folders=["/Learning/ML", "/Tools/Dev"],
    )


class TestListBookmarks:
    def test_no_filters_returns_all(self, app_ctx):
        result = _list_bookmarks_logic(app_ctx, folder="", keyword="", limit=20)
        assert "GitHub" in result
        assert "fast.ai" in result
        assert "Stack Overflow" in result

    def test_filter_by_folder(self, app_ctx):
        result = _list_bookmarks_logic(app_ctx, folder="/Learning", keyword="", limit=20)
        assert "fast.ai" in result
        assert "GitHub" not in result

    def test_filter_by_keyword(self, app_ctx):
        result = _list_bookmarks_logic(app_ctx, folder="", keyword="deep learning", limit=20)
        assert "fast.ai" in result

    def test_limit(self, app_ctx):
        result = _list_bookmarks_logic(app_ctx, folder="", keyword="", limit=1)
        assert result.count("](") == 1  # only one markdown link

    def test_no_matches(self, app_ctx):
        result = _list_bookmarks_logic(app_ctx, folder="/Nonexistent", keyword="", limit=20)
        assert "No bookmarks" in result


class TestGetBookmarkStats:
    def test_returns_stats(self, app_ctx):
        result = _get_bookmark_stats_logic(app_ctx)
        assert "Total bookmarks: 3" in result
        assert "Unique folders: 2" in result
        assert "/Tools/Dev: 2" in result


class TestSearchBookmarks:
    def test_search_returns_results(self, app_ctx):
        mock_doc = MagicMock()
        mock_doc.page_content = "GitHub\nFolder: /Tools/Dev\n\nCode hosting platform."
        mock_doc.metadata = {"source": "https://github.com", "folder": "/Tools/Dev"}

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [mock_doc]
        app_ctx.vector_store = mock_vs

        result = _search_bookmarks_logic(app_ctx, query="code hosting", k=5)
        assert "GitHub" in result
        assert "github.com" in result
        mock_vs.similarity_search.assert_called_once_with("code hosting", k=5)

    def test_search_no_results(self, app_ctx):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        app_ctx.vector_store = mock_vs

        result = _search_bookmarks_logic(app_ctx, query="nonexistent", k=5)
        assert "No bookmarks found" in result

    def test_search_clamps_k(self, app_ctx):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        app_ctx.vector_store = mock_vs

        _search_bookmarks_logic(app_ctx, query="test", k=50)
        mock_vs.similarity_search.assert_called_once_with("test", k=30)

    def test_search_clamps_k_minimum(self, app_ctx):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        app_ctx.vector_store = mock_vs

        _search_bookmarks_logic(app_ctx, query="test", k=-5)
        mock_vs.similarity_search.assert_called_once_with("test", k=1)
