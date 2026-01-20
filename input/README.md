# Input Data Directory

This directory should contain your training and test data files.

## Required Files

1. **train.csv** - Training dataset labels and metadata
2. **test.csv** - Test dataset labels and metadata
3. **dataset/** - Directory containing ELF binary files

## CSV File Format

Your CSV files should follow this structure:

```csv
file_name,md5,label,endianness,file_type,CPU,family,first_seen,size,diec_is_packed,diec_packer_info,diec_packing_method,bits,load_segments,has_section_name
abc123def...,hash123...,Malware,little endian,EXEC,x86-64,malware_family,2024-01-01,12345,False,,,64.0,2.0,True
```

### Required Columns

- **file_name**: SHA256 hash of the binary file (used to locate file in dataset/)
- **md5**: MD5 hash (for reference)
- **label**: Classification label (e.g., "Malware", "Benign")
- **family**: Malware family name (for multi-class classification)

### Optional Metadata Columns

- endianness, file_type, CPU, first_seen, size
- diec_is_packed, diec_packer_info, diec_packing_method
- bits, load_segments, has_section_name

## Dataset Directory Structure

Binary files should be organized by their hash prefix:

```
dataset/
├── 00/
│   └── 0081da7a33afa51c29b8cd11f1061d2481681321c31941b9b136e022d2536537
├── 01/
│   └── 01abcdef...
├── 02/
│   └── 02fedcba...
...
```

The first two characters of the file_name (SHA256) determine the subdirectory.

## Data Not Included

⚠️ **The actual data files are excluded from this repository for security and size reasons.**

To use this tool, you need to:

1. Prepare your own dataset of ELF binaries
2. Create CSV files with appropriate labels
3. Configure paths in `config.json`

Alternatively, use the example configuration pointing to `data/samples/` for demonstration purposes (see `config.example.json`).

## Security Warning

⚠️ **Never commit actual malware samples or CSV files containing real malware hashes to version control.**

The `.gitignore` file is configured to prevent accidental uploads.
