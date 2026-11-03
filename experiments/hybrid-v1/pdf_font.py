"""Embed the bundled unmodified DejaVu Sans TTF in a Unicode PDF (BMP cmap)."""
import functools
import pathlib
import struct
import zlib


@functools.lru_cache(maxsize=1)
def metrics():
    data=(pathlib.Path(__file__).parent/'assets/DejaVuSans.ttf').read_bytes()
    u16=lambda p:struct.unpack_from('>H',data,p)[0]
    u32=lambda p:struct.unpack_from('>I',data,p)[0]
    tables={data[p:p+4].decode('ascii'):(u32(p+8),u32(p+12)) for p in range(12,12+16*u16(4),16)}
    cmap=tables['cmap'][0]; sub=None
    for p in range(cmap+4,cmap+4+8*u16(cmap+2),8):
        at=cmap+u32(p+4)
        if u16(at)==4 and u16(p) in (0,3):sub=at
    if sub is None:raise ValueError('Police Unicode incompatible.')
    count=u16(sub+6)//2;end=sub+14;start=end+2*count+2;delta=start+2*count;ranges=delta+2*count
    mapping={}
    for i in range(count):
        for char in range(u16(start+2*i),min(u16(end+2*i)+1,65535)):
            offset=u16(ranges+2*i)
            glyph=u16(ranges+2*i+offset+2*(char-u16(start+2*i))) if offset else char
            if glyph: glyph=(glyph+u16(delta+2*i))%65536
            mapping[char]=glyph
    nmetrics=u16(tables['hhea'][0]+34);hmtx=tables['hmtx'][0];em=u16(tables['head'][0]+18)
    widths={char:round(u16(hmtx+4*min(glyph,nmetrics-1))*1000/em) for char,glyph in mapping.items()}
    return data,mapping,widths


def embed(add,stream):
    data,mapping,widths=metrics()
    file=add(stream(zlib.compress(data),b'/Filter /FlateDecode /Length1 '+str(len(data)).encode()))
    descriptor=add(('<< /Type /FontDescriptor /FontName /DejaVuSans /Flags 32 /FontBBox [-1021 -463 1793 1233] /ItalicAngle 0 /Ascent 928 /Descent -236 /CapHeight 729 /StemV 80 /FontFile2 %d 0 R >>'%file).encode())
    glyphs=bytearray(131072)
    for char,glyph in mapping.items():struct.pack_into('>H',glyphs,char*2,glyph)
    gid=add(stream(zlib.compress(glyphs),b'/Filter /FlateDecode'))
    # Identity Unicode CIDs, with widths matching the embedded font for wrapping/rendering.
    w=' '.join('%d [%d]'%(char,width) for char,width in sorted(widths.items()))
    descendant=add(('<< /Type /Font /Subtype /CIDFontType2 /BaseFont /DejaVuSans /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /FontDescriptor %d 0 R /CIDToGIDMap %d 0 R /DW 600 /W [%s] >>'%(descriptor,gid,w)).encode())
    cmap=b'/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n1 beginbfrange\n<0000> <FFFF> <0000>\nendbfrange\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend'
    unicode=add(stream(cmap))
    return add(('<< /Type /Font /Subtype /Type0 /BaseFont /DejaVuSans /Encoding /Identity-H /DescendantFonts [%d 0 R] /ToUnicode %d 0 R >>'%(descendant,unicode)).encode())


def wrap(text,size,max_width=490):
    widths=metrics()[2];lines=[]
    for word in text.split():
        current=lines[-1] if lines else ''
        if current and sum(widths.get(ord(c),600) for c in current+' '+word)*size/1000 <= max_width:
            lines[-1]+=' '+word;continue
        fragment=''
        for char in word:
            if fragment and sum(widths.get(ord(c),600) for c in fragment+char)*size/1000 > max_width:
                lines.append(fragment);fragment=''
            fragment+=char
        if fragment:lines.append(fragment)
    return lines or ['']
