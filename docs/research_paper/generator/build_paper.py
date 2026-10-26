# -*- coding: utf-8 -*-
"""Rebuild the paper's document.xml inside the original Word template.

usage: build_paper.py <template.docx or unpacked dir> <figs_dir> <out.docx> [--authors]

Content lives in paper_content.py; figures come from figs.py. Without --authors
the output is the anonymous submission (no author block, no author metadata).
"""
import copy
import importlib.util
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("paper_content", HERE / "paper_content.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

ORIG = Path(sys.argv[1])
if ORIG.suffix == ".docx":
    unpacked = Path(tempfile.mkdtemp()) / "template"
    with zipfile.ZipFile(ORIG) as z:
        z.extractall(unpacked)
    ORIG = unpacked
FIGS = Path(sys.argv[2])
OUT = Path(sys.argv[3])
WITH_AUTHORS = "--authors" in sys.argv

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
IMAGE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
HYPER_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"

MARGIN = 1100             # 0.764 in on every side (NCC 2027: at least 3/4 inch)
PAGE_W, PAGE_H = 11910, 16840
COL_SPACE = 245
TEXT_W = PAGE_W - 2 * MARGIN
COL_W = (TEXT_W - COL_SPACE) // 2
FIG_W_EMU = int(3.22 * 914400)
LINE = 240                # single line spacing

doc_tree = etree.parse(str(ORIG / "word" / "document.xml"))
root = doc_tree.getroot()
NSMAP = root.nsmap
body = root.find(f"{{{W}}}body")
originals = list(body)
NS_DECL = " ".join(f'xmlns:{k}="{v}"' for k, v in NSMAP.items() if k)


def frag(xml):
    """Parse a w:-prefixed XML fragment into an element."""
    wrapper = etree.fromstring(f"<w:root {NS_DECL}>{xml}</w:root>")
    return wrapper[0]


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def typography(text):
    text = text.replace("'", "’")
    return text


# ------------------------------------------------------------------ citations
order = []


def cite(keys):
    nums = []
    for key in keys:
        if key not in C.REFERENCES:
            raise KeyError(key)
        if key not in order:
            order.append(key)
        nums.append(order.index(key) + 1)
    nums = sorted(set(nums))
    groups, start, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        groups.append((start, prev))
        if n is not None:
            start = prev = n
    parts = []
    for a, b in groups:
        if a == b:
            parts.append(f"[{a}]")
        elif b == a + 1:
            parts.append(f"[{a}], [{b}]")
        else:
            parts.append(f"[{a}]–[{b}]")
    return ", ".join(parts)


def resolve_citations(text):
    return re.sub(r"\[(@[^\]]+)\]",
                  lambda m: cite([k.strip().lstrip("@") for k in m.group(1).split(",")]), text)


# ------------------------------------------------------------------ inline markup
def parse_inline(text, props=frozenset()):
    """-> list of (text, props). Markup: **b**, *i*, _{sub}, ^{sup}."""
    out, buf, i = [], [], 0
    italic = "i" in props

    def flush():
        if buf:
            out.append(("".join(buf), props | ({"i"} if italic else set())))
            buf.clear()

    props = frozenset(p for p in props if p != "i")
    while i < len(text):
        if text.startswith("**", i):
            j = text.index("**", i + 2)
            flush()
            out.extend(parse_inline(text[i + 2:j], props | {"b"} | ({"i"} if italic else set())))
            i = j + 2
        elif text[i] == "*":
            flush()
            italic = not italic
            i += 1
        elif text.startswith("_{", i) or text.startswith("^{", i):
            kind = "sub" if text[i] == "_" else "sup"
            depth, j = 1, i + 2
            while depth:
                depth += {"{": 1, "}": -1}.get(text[j], 0)
                j += 1
            flush()
            out.extend(parse_inline(text[i + 2:j - 1],
                                    props | {kind} | ({"i"} if italic else set())))
            i = j
        else:
            buf.append(text[i])
            i += 1
    flush()
    return out


def run_xml(text, props, size=None):
    rpr = []
    if "b" in props:
        rpr.append("<w:b/><w:bCs/>")
    if "i" in props:
        rpr.append("<w:i/><w:iCs/>")
    if "caps" in props:
        rpr.append("<w:smallCaps/>")
    if size:
        rpr.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    if "sub" in props:
        rpr.append('<w:vertAlign w:val="subscript"/>')
    if "sup" in props:
        rpr.append('<w:vertAlign w:val="superscript"/>')
    rpr_xml = f"<w:rPr>{''.join(rpr)}</w:rPr>" if rpr else ""
    pieces = text.split("\t")
    xml = []
    for k, piece in enumerate(pieces):
        if k:
            xml.append(f"<w:r>{rpr_xml}<w:tab/></w:r>")
        if piece:
            xml.append(f'<w:r>{rpr_xml}<w:t xml:space="preserve">{esc(piece)}</w:t></w:r>')
    return "".join(xml)


def runs(text, base=frozenset(), size=None):
    text = typography(resolve_citations(text))
    return "".join(run_xml(t, p, size) for t, p in parse_inline(text, frozenset(base)))


def para(ppr, content):
    return frag(f"<w:p><w:pPr>{ppr}</w:pPr>{content}</w:p>")


BODY_PPR = f'<w:pStyle w:val="BodyText"/><w:spacing w:before="0" w:line="{LINE}" w:lineRule="auto"/>'
BODY_PPR0 = (f'<w:pStyle w:val="BodyText"/><w:spacing w:before="0" w:line="{LINE}" w:lineRule="auto"/>'
             '<w:ind w:firstLine="0"/>')


def h1(text):
    return para('<w:pStyle w:val="ListParagraph"/><w:keepNext/><w:numPr><w:ilvl w:val="0"/>'
                '<w:numId w:val="5"/></w:numPr><w:spacing w:before="150" w:after="70"/>'
                '<w:ind w:left="0" w:firstLine="0"/><w:jc w:val="center"/>',
                runs(text, {"caps"}, 20))


def h2(text, num_id):
    return para(f'<w:pStyle w:val="ListParagraph"/><w:keepNext/><w:numPr><w:ilvl w:val="0"/>'
                f'<w:numId w:val="{num_id}"/></w:numPr><w:spacing w:before="90" w:after="40"/>'
                f'<w:ind w:left="288" w:hanging="288"/><w:jc w:val="left"/>',
                runs(text, {"i"}, 20))


def equation(parts, number):
    content = "\t" + "".join(parts) + "\t(" + number + ")"
    return para(f'<w:pStyle w:val="BodyText"/><w:tabs><w:tab w:val="center" w:pos="{COL_W // 2}"/>'
                f'<w:tab w:val="right" w:pos="{COL_W - 20}"/></w:tabs>'
                f'<w:spacing w:before="60" w:after="60" w:line="{LINE}" w:lineRule="auto"/>'
                '<w:ind w:left="0" w:firstLine="0"/><w:jc w:val="left"/>', runs(content))


# ------------------------------------------------------------------ figures
media = {}


def figure(name):
    rid = f"rIdFig{len(media) + 1}"
    target = f"media/{name}"
    media[rid] = (name, target)
    w, h = Image.open(FIGS / name).size
    cx = FIG_W_EMU
    cy = int(cx * h / w)
    pid = len(media)
    return frag(
        f'<w:p><w:pPr><w:pStyle w:val="BodyText"/><w:keepNext/><w:spacing w:before="100" w:after="0"/>'
        f'<w:ind w:left="0" w:firstLine="0"/><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:noProof/></w:rPr>'
        f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{pid}" name="Picture {pid}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{pid}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
        f'</wp:inline></w:drawing></w:r></w:p>')


def caption(text):
    return para('<w:spacing w:before="40" w:after="100"/><w:jc w:val="both"/>', runs(text, size=16))


def table_caption(text):
    return para('<w:keepNext/><w:spacing w:before="100" w:after="50"/><w:jc w:val="center"/>',
                runs(text, {"caps"}, 16))


def cell(text, width, align, bold=False, span=1):
    span_xml = f'<w:gridSpan w:val="{span}"/>' if span > 1 else ""
    return (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{span_xml}</w:tcPr><w:p><w:pPr>'
            f'<w:pStyle w:val="BodyText"/><w:keepNext/><w:spacing w:before="0" w:after="0" w:line="216" '
            f'w:lineRule="auto"/><w:ind w:left="0" w:firstLine="0"/><w:jc w:val="{align}"/></w:pPr>'
            f'{runs(text, {"b"} if bold else set(), 16)}</w:p></w:tc>')


def table(spec):
    widths = spec["widths"]
    total = sum(widths)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    rows = []
    header = "".join(cell(t, w, "left" if k == 0 else "center", bold=True)
                     for k, (t, w) in enumerate(zip(spec["header"], widths)))
    rows.append(f'<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>{header}</w:tr>')
    for row in spec["rows"]:
        if len(row) == 1:
            rows.append(f'<w:tr><w:trPr><w:cantSplit/></w:trPr>{cell(row[0], total, "left", span=len(widths))}</w:tr>')
        else:
            cells = "".join(cell(t, w, "left" if k == 0 else "center")
                            for k, (t, w) in enumerate(zip(row, widths)))
            rows.append(f'<w:tr><w:trPr><w:cantSplit/></w:trPr>{cells}</w:tr>')
    indent = (COL_W - total) // 2
    return frag(f'<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="{total}" w:type="dxa"/>'
                f'<w:tblInd w:w="{indent}" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
                f'<w:tblCellMar><w:left w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar>'
                f'<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" '
                f'w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(rows)}</w:tbl>')


def sect(cols_xml):
    return (f'<w:sectPr><w:type w:val="continuous"/><w:pgSz w:w="{PAGE_W}" w:h="{PAGE_H}"/>'
            f'<w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" w:bottom="{MARGIN}" w:left="{MARGIN}" '
            f'w:header="720" w:footer="720" w:gutter="0"/>{cols_xml}</w:sectPr>')


# ------------------------------------------------------------------ assemble
new = []
new.append(para('<w:pStyle w:val="BodyText"/><w:suppressAutoHyphens/><w:spacing w:before="0" w:after="120"/>'
                '<w:ind w:left="0" w:firstLine="0"/><w:jc w:val="center"/>', runs(C.TITLE, size=40)))
if WITH_AUTHORS:
    # Title section ends here; authors in three columns.
    new[-1].find(f"{{{W}}}pPr").append(frag(sect('<w:cols w:space="720"/>')))
    for k, (name, aff, city, email) in enumerate(C.AUTHORS):
        brk = '<w:r><w:br w:type="column"/></w:r>' if k else ""
        new.append(para('<w:spacing w:before="0"/><w:jc w:val="center"/>', brk + runs(name, size=18)))
        new.append(para('<w:spacing w:before="0"/><w:jc w:val="center"/>', runs(aff, {"i"}, 18)))
        new.append(para('<w:spacing w:before="0"/><w:jc w:val="center"/>', runs(city, {"i"}, 18)))
        new.append(para('<w:spacing w:before="0"/><w:jc w:val="center"/>', runs(email, size=18)))
    new[-1].find(f"{{{W}}}pPr").append(frag(sect('<w:cols w:num="3" w:space="300"/>')))
# Closes the single-column title section.
spacer = para('<w:spacing w:before="0" w:after="60"/><w:ind w:left="0" w:firstLine="0"/>', "")
spacer.find(f"{{{W}}}pPr").append(frag(sect('<w:cols w:space="720"/>')))
new.append(spacer)

new.append(para('<w:spacing w:before="0" w:after="0"/><w:ind w:left="0" w:firstLine="272"/><w:jc w:val="both"/>',
                run_xml("Abstract", {"b", "i"}, 18) + run_xml("—", {"b", "i"}, 18)
                + runs(C.ABSTRACT, {"b"}, 18)))
new.append(para('<w:spacing w:before="120" w:after="0"/><w:ind w:left="0" w:firstLine="272"/><w:jc w:val="both"/>',
                run_xml("Keywords", {"b", "i"}, 18) + run_xml("—", {"b", "i"}, 18)
                + runs(C.KEYWORDS, {"b", "i"}, 18)))

for block in C.BODY:
    kind = block[0]
    if kind == "h1":
        new.append(h1(block[1]))
    elif kind == "h2a":
        new.append(h2(block[1], 4))
    elif kind == "h2b":
        new.append(h2(block[1], 2))
    elif kind == "p":
        new.append(para(BODY_PPR, runs(block[1])))
    elif kind == "p0":
        new.append(para(BODY_PPR0, runs(block[1])))
    elif kind == "run":
        new.append(para(BODY_PPR, runs("*" + block[1] + "* " + block[2])))
    elif kind == "eq":
        new.append(equation(block[1], block[2]))
    elif kind == "fig":
        new.append(figure(block[1]))
    elif kind == "cap":
        new.append(caption(block[1]))
    elif kind == "tcap":
        new.append(table_caption(block[1]))
    elif kind == "table":
        new.append(table(C.TABLES[block[1]]))
    else:
        raise ValueError(kind)

new.append(para('<w:pStyle w:val="BodyText"/><w:keepNext/><w:spacing w:before="150" w:after="60"/>'
                '<w:ind w:left="0" w:firstLine="0"/><w:jc w:val="center"/>', runs("References", {"caps"})))
for key in order:
    new.append(para('<w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
                    '<w:spacing w:before="20" w:line="228" w:lineRule="auto"/><w:ind w:left="360" w:hanging="360"/>'
                    '<w:jc w:val="both"/>', runs(C.REFERENCES[key], size=16)))
unused = [k for k in C.REFERENCES if k not in order]
if unused:
    raise SystemExit(f"uncited references: {unused}")

for child in originals:
    body.remove(child)
for el in new:
    body.append(el)
body.append(frag(sect(f'<w:cols w:num="2" w:space="{COL_SPACE}"/>')))

# ------------------------------------------------------------------ package
work = Path(tempfile.mkdtemp())
pkg = work / "pkg"
shutil.copytree(ORIG, pkg)
for old in (pkg / "word" / "media").glob("*"):
    old.unlink()
for rid, (name, target) in media.items():
    shutil.copy(FIGS / name, pkg / "word" / target)

rels_path = pkg / "word" / "_rels" / "document.xml.rels"
rels = etree.parse(str(rels_path))
for rel in list(rels.getroot()):
    if rel.get("Type") in (IMAGE_REL, HYPER_REL):
        rels.getroot().remove(rel)
for rid, (name, target) in media.items():
    etree.SubElement(rels.getroot(), f"{{{PKG_REL}}}Relationship",
                     Id=rid, Type=IMAGE_REL, Target=target)
rels.write(str(rels_path), xml_declaration=True, encoding="UTF-8", standalone=True)

doc_tree.write(str(pkg / "word" / "document.xml"), xml_declaration=True, encoding="UTF-8", standalone=True)

# Heading numbers: "I. Introduction" centred, number and text separated by a space.
num_path = pkg / "word" / "numbering.xml"
num = etree.parse(str(num_path))
for absnum in num.getroot().findall(f"{{{W}}}abstractNum"):
    if absnum.get(f"{{{W}}}abstractNumId") == "2":
        lvl = absnum.find(f"{{{W}}}lvl")
        for old in lvl.findall(f"{{{W}}}suff"):
            lvl.remove(old)
        suff = etree.Element(f"{{{W}}}suff")
        suff.set(f"{{{W}}}val", "space")
        lvl.find(f"{{{W}}}lvlText").addprevious(suff)
        lvl.find(f"{{{W}}}lvlJc").set(f"{{{W}}}val", "left")
        ppr = lvl.find(f"{{{W}}}pPr")
        for child in list(ppr):
            ppr.remove(child)
        ppr.append(frag('<w:ind w:left="0" w:firstLine="0"/>'))
num.write(str(num_path), xml_declaration=True, encoding="UTF-8", standalone=True)

# Metadata: nothing identifying, title only.
core_path = pkg / "docProps" / "core.xml"
core = core_path.read_text(encoding="utf-8")
core = re.sub(r"<dc:title>.*?</dc:title>|<dc:title/>", "<dc:title/>", core)
core_path.write_text(core, encoding="utf-8")
app_path = pkg / "docProps" / "app.xml"
app = app_path.read_text(encoding="utf-8")
app = re.sub(r"<vt:lpstr>[^<]*</vt:lpstr>", "<vt:lpstr></vt:lpstr>", app)
app = re.sub(r"<Company>[^<]*</Company>", "<Company></Company>", app)
app_path.write_text(app, encoding="utf-8")

OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    ct = pkg / "[Content_Types].xml"
    z.write(ct, "[Content_Types].xml")
    for path in sorted(pkg.rglob("*")):
        if path.is_file() and path != ct:
            z.write(path, path.relative_to(pkg).as_posix())
shutil.rmtree(work)
print(f"wrote {OUT} with {len(order)} references, {len(media)} figures")
