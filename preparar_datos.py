import os, shutil

possible_roots = [
    os.path.expanduser('~/Downloads'),
    os.path.expanduser('~/Descargas'),
    'C:/Users/benja/Downloads',
    'C:/Users/benja/Descargas'
]

source_dir = None
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

for root_path in possible_roots:
    if os.path.exists(root_path):
        for root, dirs, files in os.walk(root_path):
            if all(c in dirs for c in classes):
                source_dir = root
                break
        if source_dir:
            break

if not source_dir:
    print("ERROR: No se encontro ninguna carpeta con las subcarpetas de residuos en Descargas.")
else:
    print(f"Carpeta de origen encontrada en: {source_dir}")
    train_base = 'data/test_dataset/train'
    val_base = 'data/test_dataset/validation'

    if os.path.exists('data/test_dataset'):
        shutil.rmtree('data/test_dataset')

    total_copied = 0
    for cls in classes:
        src_cls = os.path.join(source_dir, cls)
        files = [f for f in os.listdir(src_cls) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        split = int(len(files) * 0.8)
        
        os.makedirs(os.path.join(train_base, cls), exist_ok=True)
        os.makedirs(os.path.join(val_base, cls), exist_ok=True)
        
        for i, f in enumerate(files):
            src_file = os.path.join(src_cls, f)
            if i < split:
                shutil.copy(src_file, os.path.join(train_base, cls, f))
            else:
                shutil.copy(src_file, os.path.join(val_base, cls, f))
            total_copied += 1

    print(f"EXITO: Se copiaron {total_copied} imagenes reales en data/test_dataset.")
