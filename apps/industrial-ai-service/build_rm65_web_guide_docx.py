#!/usr/bin/env python3
"""Convert the locally authored RM65 HTML guide into a simple image-embedded DOCX.

This intentionally uses only packages already present on the shared machine and
does not call LibreOffice or download dependencies.
"""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
import shutil
import tempfile

from bs4 import BeautifulSoup, NavigableString, Tag
from PIL import Image


ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2]))
HTML = ROOT / "workspace/docs/机械臂web/RM65B_WEB示教器与O7灵巧手联动攻略_v1.html"
OUTPUT = ROOT / "workspace/docs/机械臂web/RM65B_WEB示教器与O7灵巧手联动攻略_v1.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"


def paragraph(text, style=None, bold=False, mono=False, page_break=False):
    text = " ".join(text.split()) if not mono else text
    ppr = ""
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if page_break:
        ppr += '<w:pageBreakBefore/>'
    ppr = f"<w:pPr>{ppr}</w:pPr>" if ppr else ""
    rpr = ""
    if bold:
        rpr += "<w:b/>"
    if mono:
        rpr += '<w:rFonts w:ascii="DejaVu Sans Mono" w:hAnsi="DejaVu Sans Mono"/>'
    rpr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    preserve = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    return f"<w:p>{ppr}<w:r>{rpr}<w:t{preserve}>{escape(text)}</w:t></w:r></w:p>"


def image_paragraph(rid, image_path, index):
    with Image.open(image_path) as im:
        width_px, height_px = im.size
    max_width = 6.35
    max_height = 8.0
    width_in = min(max_width, width_px / 110.0)
    height_in = width_in * height_px / width_px
    if height_in > max_height:
        height_in = max_height
        width_in = height_in * width_px / height_px
    cx, cy = int(width_in * 914400), int(height_in * 914400)
    return f'''<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>
<wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="{WP}">
<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{index}" name="截图 {index}"/>
<a:graphic xmlns:a="{A}"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic xmlns:pic="{PIC}"><pic:nvPicPr><pic:cNvPr id="{index}" name="image{index}.png"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rid}" xmlns:r="{R}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'''


def table_xml(table):
    rows = []
    for tr in table.find_all("tr", recursive=False):
        cells = []
        for cell in tr.find_all(["th", "td"], recursive=False):
            text = " ".join(cell.get_text(" ", strip=True).split())
            shade = '<w:shd w:fill="E8F1F7"/>' if cell.name == "th" else ""
            cells.append(f'<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/>{shade}</w:tcPr>{paragraph(text, bold=cell.name == "th")}</w:tc>')
        rows.append("<w:tr>" + "".join(cells) + "</w:tr>")
    return '<w:tbl><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr>' + "".join(rows) + "</w:tbl>"


def content_xml(soup, temp_dir):
    parts, rels, media = [], [], []
    image_index = 0

    def add_tag(node):
        nonlocal image_index
        if not isinstance(node, Tag):
            return
        if node.name in {"h1", "h2", "h3"}:
            parts.append(paragraph(node.get_text(" ", strip=True), {"h1":"Title","h2":"Heading1","h3":"Heading2"}[node.name], page_break="page" in node.get("class", [])))
        elif node.name == "p":
            style = "Caption" if "cap" in node.get("class", []) else None
            parts.append(paragraph(node.get_text(" ", strip=True), style))
        elif node.name in {"ul", "ol"}:
            for number, li in enumerate(node.find_all("li", recursive=False), 1):
                prefix = f"{number}. " if node.name == "ol" else "• "
                parts.append(paragraph(prefix + li.get_text(" ", strip=True), "ListParagraph"))
        elif node.name == "pre":
            parts.append(paragraph(node.get_text(), "Code", mono=True))
        elif node.name == "table":
            parts.append(table_xml(node))
        elif node.name == "img":
            source = (HTML.parent / node["src"]).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            image_index += 1
            media_name = f"image{image_index}{source.suffix.lower()}"
            copied = temp_dir / "word/media" / media_name
            shutil.copy2(source, copied)
            rid = f"rIdImage{image_index}"
            rels.append((rid, f"media/{media_name}"))
            media.append(copied)
            parts.append(image_paragraph(rid, source, image_index))
        elif node.name == "div":
            text = " ".join(node.get_text(" ", strip=True).split())
            parts.append(paragraph(text, "Callout", bold=True))

    for child in soup.body.children:
        if isinstance(child, NavigableString) and not child.strip():
            continue
        add_tag(child)
    return parts, rels, media


def write_package():
    soup = BeautifulSoup(HTML.read_text(encoding="utf-8"), "html.parser")
    with tempfile.TemporaryDirectory(prefix="rm65_docx_", dir="/tmp") as temp:
        temp_dir = Path(temp)
        (temp_dir / "_rels").mkdir()
        (temp_dir / "word/_rels").mkdir(parents=True)
        (temp_dir / "word/media").mkdir(parents=True)
        parts, image_rels, media = content_xml(soup, temp_dir)

        document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>{''.join(parts)}
<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="900" w:right="900" w:bottom="900" w:left="900"/></w:sectPr>
</w:body></w:document>'''
        styles = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="21"/></w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:pPr><w:spacing w:after="100" w:line="330" w:lineRule="auto"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="220"/><w:jc w:val="center"/></w:pPr><w:rPr><w:b/><w:color w:val="154B7D"/><w:sz w:val="38"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="300" w:after="140"/><w:keepNext/></w:pPr><w:rPr><w:b/><w:color w:val="155B8F"/><w:sz w:val="29"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="220" w:after="100"/><w:keepNext/></w:pPr><w:rPr><w:b/><w:color w:val="276B50"/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:keepNext/></w:pPr><w:rPr><w:i/><w:color w:val="58626C"/><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="360" w:hanging="180"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:pPr><w:shd w:fill="F3F5F7"/><w:ind w:left="180" w:right="180"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Callout"><w:name w:val="Callout"/><w:basedOn w:val="Normal"/><w:pPr><w:shd w:fill="EDF7FF"/><w:ind w:left="180" w:right="180"/><w:spacing w:before="100" w:after="140"/></w:pPr></w:style>
</w:styles>'''
        content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''
        root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
        document_rels = ['''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>''']
        document_rels.extend(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>' for rid, target in image_rels)
        document_rels.append("</Relationships>")

        files = {
            "[Content_Types].xml": content_types,
            "_rels/.rels": root_rels,
            "word/document.xml": document,
            "word/styles.xml": styles,
            "word/_rels/document.xml.rels": "".join(document_rels),
        }
        for relative, data in files.items():
            path = temp_dir / relative
            path.write_text(data, encoding="utf-8")

        with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
            for path in sorted(temp_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(temp_dir))
    print(f"created={OUTPUT}")
    print(f"embedded_images={len(media)}")


if __name__ == "__main__":
    write_package()
