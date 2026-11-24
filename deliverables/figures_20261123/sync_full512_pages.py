"""Copy the full-512 D7 section and figures into the existing GitHub Pages page.

Requires mistune. Run from any directory after editing deliverable_7.md.
Only the marked full-512 section and its matching scope notes are updated.
"""
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil

import mistune

REPO = Path(__file__).resolve().parents[2]
source = REPO/'deliverables/deliverable_7.md'
target = REPO/'docs/deliverable7/index.html'
text = source.read_text()
start = text.index('## BCSS full-resolution correction experiments')
if '<!-- END FULL512 BCSS -->' in text[start:]:
    end = text.index('<!-- END FULL512 BCSS -->', start)
else:
    following = re.search(r'\n## [^#]', text[start + 1:])
    end = start + 1 + following.start() if following else len(text)
section = text[start:end].rstrip().removesuffix('---').rstrip()
render = mistune.create_markdown(escape=True, plugins=['table'])
html = render(section)
# This is a subsection of BCSS Results (h3), rather than a new dataset section.
html = html.replace('<h2>', '<h4>').replace('</h2>', '</h4>')
html = html.replace('<h3>', '<h5>').replace('</h3>', '</h5>')
prefix = 'https://github.com/urvib3/csci-401-capstone-histopathology/blob/main/'
html = html.replace('href="../experiments/', f'href="{prefix}experiments/')
for name in re.findall(r'src="figures_20261123/([^\"]+)"', html):
    shutil.copy2(source.parent/'figures_20261123'/name, target.parent/name)
html = html.replace('src="figures_20261123/', 'src="')
block = '<!-- BEGIN FULL512 BCSS -->\n<section id="full512">\n' + html + '</section>\n<!-- END FULL512 BCSS -->\n\n'
page = re.sub(r'<!-- BEGIN FULL512 BCSS -->.*?<!-- END FULL512 BCSS -->\n*', '',
              target.read_text(), flags=re.S)


class BCSSSection(HTMLParser):
    """Locate the closing tag of BCSS, including any nested sections."""

    depth = 0
    closing_position = None

    def handle_starttag(self, tag, attrs):
        if tag == 'section':
            if self.depth or dict(attrs).get('id') == 'bcss':
                self.depth += 1

    def handle_endtag(self, tag):
        if tag == 'section' and self.depth:
            self.depth -= 1
            if self.depth == 0:
                self.closing_position = self.getpos()


location = BCSSSection()
location.feed(page)
if location.closing_position is None:
    raise ValueError('Could not locate the BCSS section closing tag')
line, column = location.closing_position
offset = sum(len(part) for part in page.splitlines(keepends=True)[:line - 1]) + column
page = page[:offset] + '\n' + block + page[offset:]
if '<a href="#full512">' not in page:
    page = page.replace('</nav>', '<a href="#full512">Full-512 BCSS</a></nav>', 1)
note = '<p class="note" id="full512-scope">The BCSS recap uses the earlier pretrained U-Net and central-256 field. The trained full-512 checkpoint and its correction results are <a href="#full512">reported separately below</a>.</p>'
if 'id="full512-scope"' not in page:
    page = page.replace('<table>', note + '\n\n<table>', 1)
page = page.replace("Ryan's full-resolution U-Net · 185 dev tiles · no training.",
                    'Frozen trained full-resolution U-Net · 185 dev tiles · post-processing only.')
target.write_text(page)
print('Updated', target)
