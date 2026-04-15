
from pathlib import Path

f = Path('data/drug-smiles.xlsx').resolve()
print('绝对路径 :', f)
print('存在?    :', f.exists())
print('大小     :', f.stat().st_size, 'bytes')

head = f.read_bytes()[:8]
print('头 8 字节:', head)

# 快速判型
if head.startswith(b'PK'):
    print('→ 这是 ZIP 格式（真 xlsx 属于此类）✅')
elif head.startswith(b'{"smiles"') or head.startswith(b'smiles'):
    print('→ 这是纯文本/CSV ❌ 请改 pd.read_csv')
elif len(head) == 0:
    print('→ 文件空 ❌')
else:
    print('→ 未知格式，可能损坏或 HTML ❌')