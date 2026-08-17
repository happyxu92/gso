import subprocess
import sys

from gso.harness.environment.patches import apply_patch_requests


def test_cached_requests_response_supports_streaming(tmp_path):
    test_code = """
url = 'https://example.test/cached.bin'
cached_content = b'abcdef'
with open(url_to_filename(url), 'wb') as cache_file:
    cache_file.write(cached_content)

response = requests.get(url)
assert response._content_consumed is True
assert response.headers['X-From-Cache'] == 'true'
assert list(response.iter_content(chunk_size=2)) == [b'ab', b'cd', b'ef']
response.close()
"""
    patched_test = apply_patch_requests(test_code).replace(
        "CACHE_DIR = '/.url_cache'",
        f"CACHE_DIR = {str(tmp_path)!r}",
    )

    subprocess.run(
        [sys.executable, "-c", patched_test],
        check=True,
        capture_output=True,
        text=True,
    )
