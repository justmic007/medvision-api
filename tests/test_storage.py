"""Tests for object storage (Phase 6). The S3 client is mocked so these tests
don't require a running MinIO."""
from unittest.mock import MagicMock, patch


def test_upload_scan_returns_key():
    import app.services.storage as storage
    with patch.object(storage, "_client") as mock_client:
        mock_client.return_value = MagicMock()
        key = storage.upload_scan(b"bytes", ".jpg")
    assert key.startswith("scans/")
    assert key.endswith(".jpg")


def test_download_scan_returns_bytes():
    import app.services.storage as storage
    fake = MagicMock()
    fake.get_object.return_value = {"Body": MagicMock(read=lambda: b"scan-bytes")}
    with patch.object(storage, "_client", return_value=fake):
        data = storage.download_scan("scans/abc.jpg")
    assert data == b"scan-bytes"


def test_upload_uses_configured_bucket():
    import app.services.storage as storage
    fake = MagicMock()
    with patch.object(storage, "_client", return_value=fake):
        storage.upload_scan(b"x", ".png")
    _, kwargs = fake.put_object.call_args
    assert kwargs["Bucket"] == storage._settings.s3_bucket
    assert kwargs["Key"].endswith(".png")
