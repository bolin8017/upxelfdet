# Example Data

This directory contains minimal example data for demonstration purposes.

## Directory Structure

```
data/
├── samples/
│   ├── 00/              # Example feature and vectorization output
│   └── ...
└── README.md
```

## Getting Full Dataset

The complete dataset used in this research is not included in this repository due to:
- Large file sizes
- Security considerations (contains malware samples)
- Dataset licensing restrictions

To obtain the full dataset or use your own data:

1. **Prepare your dataset** in the following format:
   - Place ELF binary files in `input/dataset/` organized by hash prefixes (e.g., `input/dataset/00/`, `input/dataset/01/`, etc.)
   - Create CSV files with columns: `file_name`, `md5`, `label`, and other metadata

2. **CSV Format Example**:
   ```csv
   file_name,md5,label,endianness,file_type,CPU,family,first_seen,size,diec_is_packed,diec_packer_info,diec_packing_method,bits,load_segments,has_section_name
   abc123...,def456...,Malware,little endian,EXEC,x86-64,family_name,2024-01-01,12345,False,,,64.0,2.0,True
   ```

3. **Configure paths** in `config.json`:
   - `data.train`: Path to training CSV
   - `data.test`: Path to test CSV
   - `data.dataset`: Path to dataset directory containing binaries

## Sample Data

The `samples/` directory contains examples of:
- **Feature extraction output**: Binary section features extracted from ELF files
- **Vectorization output**: N-gram based feature vectors

These examples demonstrate the intermediate data format used by the detector.

## Security Warning

⚠️ **Do not share or upload actual malware samples to public repositories.**

This example data contains only processed features and vectors, not actual malware binaries.
