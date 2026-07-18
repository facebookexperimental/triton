from nightly_gen_simple_index import render_project_index, render_root_index


def test_project_index_lists_files_with_hash():
    html = render_project_index([
        ("fbtriton-3.8.0.dev20260717-cp312-cp312-manylinux_2_28_x86_64.whl", "https://example/dl/fbtriton-...whl",
         "deadbeef"),
    ])
    assert "fbtriton-3.8.0.dev20260717-cp312" in html
    assert "#sha256=deadbeef" in html
    assert "<a href=" in html


def test_root_index_links_project():
    html = render_root_index()
    assert 'href="fbtriton/"' in html
