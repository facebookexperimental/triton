import unittest
from html.parser import HTMLParser

from generate_tlx_site import render_markdown


class ImageCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.images.append(dict(attrs))


def images_in(source, **kwargs):
    parser = ImageCollector()
    parser.feed(render_markdown(source, **kwargs))
    return parser.images


class MarkdownImageTest(unittest.TestCase):
    def test_relative_image_uses_markdown_source_directory(self):
        image, = images_in('![Entry initialization](../assets/cluster.svg)')
        self.assertEqual(image["src"], "content/../assets/cluster.svg")
        self.assertEqual(image["alt"], "Entry initialization")

    def test_root_page_uses_website_content_directory(self):
        image, = images_in('![Entry initialization](../assets/cluster.svg)', from_root=True)
        self.assertEqual(image["src"], "website/content/../assets/cluster.svg")

    def test_external_image_preserves_url_and_escapes_attributes(self):
        url = "https://example.com/diagram.svg?a=1&b=2"
        image, = images_in(f'![A "barrier" & owner]({url})')
        self.assertEqual(image["src"], url)
        self.assertEqual(image["alt"], 'A "barrier" & owner')

    def test_absolute_image_preserves_path(self):
        image, = images_in('![Entry initialization](/assets/cluster.svg)')
        self.assertEqual(image["src"], "/assets/cluster.svg")

    def test_image_syntax_inside_code_remains_literal(self):
        self.assertEqual(images_in('```text\n![Not an image](diagram.svg)\n```'), [])

    def test_table_after_image_is_rendered(self):
        html = render_markdown(
            '![Entry initialization](../assets/cluster.svg)\n\n'
            '| Requirement | Owner |\n|---|---|\n| Initialization | Compiler |\n'
        )
        self.assertIn("<img ", html)
        self.assertIn("<td>Initialization</td><td>Compiler</td>", html)


if __name__ == "__main__":
    unittest.main()
