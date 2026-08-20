import pytest

from gso.data.dataset import create_instance_id


@pytest.mark.parametrize(
    ("api", "api_slug"),
    [
        ("DataFrame.__setitem__", "dataframe-setitem"),
        ("NoBadWordsLogitsProcessor.__call__", "nobadwordslogitsprocessor-call"),
        ("module::function(value, other)", "module-function-value-other"),
        ("API / name-with spaces", "api-name-with-spaces"),
    ],
)
def test_create_instance_id_includes_sanitized_api(api, api_slug):
    assert create_instance_id("owner/repo", api, "abcdef1") == (
        f"owner__repo-{api_slug}-abcdef1"
    )


def test_create_instance_id_rejects_api_without_alphanumeric_characters():
    with pytest.raises(ValueError, match="API name cannot be converted"):
        create_instance_id("owner/repo", "._::()", "abcdef1")
