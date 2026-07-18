"""Generate a minimal PEP 503 'simple' index (HTML) for GitHub Pages."""
import html as _html

_PROJECT_TMPL = ("<!DOCTYPE html><html><head>"
                 "<meta name='pypi:repository-version' content='1.0'>"
                 "<title>Links for fbtriton</title></head><body>"
                 "<h1>Links for fbtriton</h1>\n{anchors}\n</body></html>\n")


def render_project_index(files):
    anchors = "\n".join('<a href="{url}#sha256={sha}">{name}</a><br>'.format(url=_html.escape(url), sha=_html.escape(
        sha), name=_html.escape(name)) for name, url, sha in files)
    return _PROJECT_TMPL.format(anchors=anchors)


def render_root_index():
    return ("<!DOCTYPE html><html><body>"
            '<a href="fbtriton/">fbtriton</a><br>'
            "</body></html>\n")
