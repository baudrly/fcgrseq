# -*- coding: utf-8 -*-
"""
Data Acquisition: Fetching, Parsing, Cleaning Genomic Sequences.
Enhanced with genome sampler compatibility and performance optimizations.
"""
import time
import logging
import hashlib
from io import StringIO
import re
from typing import Dict, List, Tuple, Optional, Union
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Import necessary config values or pass them as arguments
from .config import (
    NCBI_FETCH_DELAY, MIN_SEQ_LEN, MAX_SEQ_LEN, VALID_BASES_BYTES,
    NCBI_DB, NCBI_RETTYPE_FASTA, NCBI_RETTYPE_GB, NCBI_RETMODE
)
from .utils import setup_requests_session # Use the utility for session setup

# Module-level cache object (initialized by pipeline/cli)
memory_cache = None

# Compiled regex patterns for performance
GENOME_SAMPLER_PATTERN = re.compile(r'^>([^|]+)\|([^|]+)\|(.*)$')
ACCESSION_PATTERN = re.compile(r'([A-Z]{1,2}_?\d+(?:\.\d+)?)')

def configure_entrez(email, api_key, session):
    """Configures Biopython Entrez settings."""
    if not email or email == "your.email@example.com":
        logging.error("CRITICAL ERROR: Entrez email is not set or is default. Please configure `ENTREZ_EMAIL` in config.py or provide via CLI.")
        raise ValueError("Entrez email not configured.")

    Entrez.email = email
    Entrez.tool = "FCGRAnalyzer/1.0" # Identify the tool
    if api_key:
        Entrez.api_key = api_key
        logging.info("NCBI API Key provided. Rate limit potentially higher (e.g., 10 req/sec).")
    else:
        logging.warning(f"NCBI API Key not provided. Using default rate limit (e.g., 3 req/sec). Fetch delay set to {NCBI_FETCH_DELAY}s.")

    # Use the pre-configured session from utils
    if session:
        Entrez.Session = session
        logging.debug("Entrez configured with custom requests session and retry logic.")
    else:
        logging.warning("No custom requests session provided to Entrez. Using default.")

# Internal fetching function - potentially cached
def _fetch_ncbi_record_uncached(identifier, db, rettype, retmode):
    """Actual NCBI fetching logic using Entrez.efetch (Uncached)."""
    logging.debug(f"NCBI Fetch: id={identifier}, db={db}, rettype={rettype}, retmode={retmode}")
    handle = None
    record_data = None
    try:
        handle = Entrez.efetch(db=db, id=identifier, rettype=rettype, retmode=retmode)
        record_data = handle.read()
        handle.close() # Close handle immediately after reading

        # Decode if necessary
        if isinstance(record_data, bytes):
            try:
                record_data = record_data.decode('utf-8', errors='ignore')
            except UnicodeDecodeError:
                logging.warning(f"Could not decode NCBI response for {identifier} as UTF-8. Treating as raw.")

        # Basic validation of response content
        if not record_data or "Error" in record_data[:200] or "Failed" in record_data[:200] or "<error>" in record_data[:200].lower():
             logging.warning(f"Invalid data or error received from NCBI for {identifier}. Response head: {str(record_data)[:500]}...")
             raise ValueError(f"Invalid data/error from NCBI for {identifier}")

        logging.debug(f"Fetch successful for {identifier}. Length: {len(record_data)}")
        # Apply delay _after successful fetch_
        time.sleep(NCBI_FETCH_DELAY)
        return record_data

    except Exception as e:
        logging.error(f"Entrez.efetch failed for {identifier}: {type(e).__name__}: {e}", exc_info=False)
        if handle: # Ensure handle is closed even on error
            try:
                handle.close()
            except Exception as he:
                logging.error(f"Error closing handle for {identifier} after fetch failure: {he}")
        return None

def fetch_ncbi_record(identifier, db, rettype, retmode):
    """
    Fetches data from NCBI, using the cache if available.
    """
    if memory_cache:
        logging.debug(f"Cache lookup/call for: id={identifier}, db={db}, rettype={rettype}")
        return memory_cache.cache(_fetch_ncbi_record_uncached)(identifier, db, rettype, retmode)
    else:
        # Call the uncached function directly if caching is disabled
        return _fetch_ncbi_record_uncached(identifier, db, rettype, retmode)

def parse_sequence_data(identifier, record_data, rettype):
    """Parses fetched data (string) into a Bio.SeqRecord."""
    if not record_data:
        logging.debug(f"Parsing skipped for {identifier}: No data provided.")
        return None

    logging.debug(f"Attempting to parse {identifier} as {rettype} format...")
    try:
        # Use StringIO to simulate a file handle
        handle = StringIO(record_data)
        records = list(SeqIO.parse(handle, rettype))

        if not records:
            logging.warning(f"Parsing failed for {identifier}: No {rettype.upper()} records found in data.")
            logging.debug(f"Data received (first 500 chars): {record_data[:500]}...")
            return None

        # Use the first record if multiple are present (common in FASTA)
        record = records[0]
        if len(records) > 1:
            logging.warning(f"Multiple {rettype.upper()} records found for {identifier}. Using the first one (ID: {record.id}).")

        # Basic validation of the parsed record
        if not hasattr(record, 'seq') or record.seq is None:
             logging.warning(f"Parsing issue for {identifier}: Parsed record lacks a valid 'seq' attribute.")
             return None
        if len(record.seq) == 0:
            logging.warning(f"Parsing successful for {identifier}, but sequence is empty. Returning None.")
            return None

        # Standardize description (often contains more info than id)
        record.description = record.description or record.id # Use description if available, else id
        logging.debug(f"Parsing successful for {identifier}. Record ID: {record.id}, Seq Length: {len(record.seq)}")
        return record

    except Exception as e:
        logging.error(f"Parsing failed for {identifier} (format: {rettype}): {type(e).__name__}: {e}", exc_info=False)
        logging.debug(f"Data causing parsing error (first 500 chars): {record_data[:500]}...")
        return None

def get_feature_sequence(gb_record: SeqRecord, feature_type: str, feature_qualifiers: dict = None, index: int = 0) -> Optional[SeqRecord]:
    """
    Extracts a specific feature sequence from a GenBank record.
    """
    if gb_record is None or not hasattr(gb_record, 'features'):
        logging.debug("Feature extraction skipped: Invalid or featureless GenBank record provided.")
        return None

    logging.debug(f"Searching for feature type='{feature_type}' (qualifiers={feature_qualifiers}, index={index}) in record {gb_record.id}")

    found_features = []
    ft_lower = feature_type.lower()

    for feature in gb_record.features:
        if feature.type.lower() != ft_lower:
            continue

        # Check qualifiers if specified
        match = True
        if feature_qualifiers:
            for qk, qv_list in feature_qualifiers.items():
                qk_lower = qk.lower()
                # Find the actual qualifier key case-insensitively
                fq_keys_lower = {k.lower(): k for k in feature.qualifiers}
                if qk_lower not in fq_keys_lower:
                    match = False
                    break # This qualifier is missing entirely

                # Get actual values, ensure it's a list
                actual_key = fq_keys_lower[qk_lower]
                actual_values = feature.qualifiers.get(actual_key, [])
                if not isinstance(actual_values, list):
                    actual_values = [actual_values]

                # Ensure desired values is also a list
                desired_values = qv_list if isinstance(qv_list, list) else [qv_list]

                # Check if ANY desired value is found (case-insensitive substring) within ANY actual value
                if not any(str(desired).lower() in str(actual).lower()
                           for desired in desired_values
                           for actual in actual_values):
                    match = False
                    break # Desired value not found for this qualifier

        if match:
            found_features.append(feature)

    if not found_features:
        logging.debug(f"No features matched type='{feature_type}' with specified qualifiers in {gb_record.id}.")
        return None

    if index >= len(found_features):
        logging.warning(f"Feature index {index} is out of bounds. Found {len(found_features)} matching features for type='{feature_type}' in {gb_record.id}. Returning None.")
        return None

    target_feature = found_features[index]
    logging.debug(f"Extracting sequence for matched feature at index {index}: Location={target_feature.location}")

    try:
        # Extract sequence using the feature's location
        extracted_sequence = target_feature.extract(gb_record.seq)

        if extracted_sequence is None:
             logging.warning(f"Feature extraction returned None for type='{feature_type}', index={index} in {gb_record.id}. Location: {target_feature.location}")
             return None
        if len(extracted_sequence) == 0:
            logging.warning(f"Extracted sequence for feature type='{feature_type}', index={index} in {gb_record.id} is empty. Location: {target_feature.location}")

        # Create a new SeqRecord for the extracted feature
        feature_id = f"{gb_record.id}|{feature_type}|{target_feature.location.start}_{target_feature.location.end}|{index}"
        feature_desc = f"{feature_type} (index {index}) extracted from {gb_record.id}. Loc: {target_feature.location}. Quals: {dict(target_feature.qualifiers)}"
        feature_record = SeqRecord(seq=extracted_sequence, id=feature_id, description=feature_desc[:200]) # Limit desc length

        logging.debug(f"Successfully extracted feature sequence. New ID: {feature_id}, Length: {len(feature_record.seq)}")
        return feature_record

    except Exception as e:
        logging.error(f"Error extracting sequence for feature type='{feature_type}', index={index} from {gb_record.id} (Location: {target_feature.location}): {type(e).__name__}: {e}", exc_info=False)
        logging.debug(f"Problematic feature details: {target_feature}")
        return None

def clean_sequence(sequence: str, original_id: str) -> Optional[str]:
    """
    Cleans a DNA sequence: converts to uppercase, removes non-ATGC chars.
    Optimized with numpy for better performance on large sequences.
    """
    if not isinstance(sequence, str):
        logging.warning(f"Invalid input sequence for cleaning (ID: {original_id}). Type: {type(sequence)}. Skipping.")
        return None

    if not sequence: # Handle empty string explicitly after type check
        return ""

    seq_upper = sequence.upper()

    # More robust cleaning using bytes and translate
    try:
        seq_bytes = seq_upper.encode('ascii', errors='replace') # Replace non-ASCII with '?'
        # Define valid bytes (A, T, G, C)
        valid_map = bytes.maketrans(VALID_BASES_BYTES, VALID_BASES_BYTES)
        # Identify all bytes that are NOT A, T, G, C
        invalid_bytes = bytes(set(range(256)) - set(VALID_BASES_BYTES))
        # Remove the invalid bytes
        cleaned_bytes = seq_bytes.translate(valid_map, invalid_bytes)
        final_seq_str = cleaned_bytes.decode('ascii') # Should now only contain ATGC

        cleaned_len = len(final_seq_str)
        original_len = len(sequence)
        removed_count = original_len - cleaned_len
        if removed_count > 0:
             logging.debug(f"Cleaned sequence {original_id}: Removed {removed_count} non-ATGC characters. Original len: {original_len}, Cleaned len: {cleaned_len}")
        else:
             logging.debug(f"Sequence {original_id} contained only ATGC. Length: {cleaned_len}")

        return final_seq_str

    except Exception as e:
        logging.error(f"Error during sequence cleaning for {original_id}: {e}", exc_info=False)
        return None

def parse_genome_sampler_header(header_line: str) -> Optional[Dict[str, str]]:
    """
    Parses headers from genome sampler output format.
    Format: >species|biotype|description
    Returns dict with species, biotype, description fields.
    """
    if not header_line or not header_line.startswith('>'):
        return None
    
    # Remove '>' and match pattern
    match = GENOME_SAMPLER_PATTERN.match(header_line)
    if match:
        species, biotype, description = match.groups()
        # Clean up species name (replace underscores with spaces for display)
        species_display = species.replace('_', ' ')
        
        # Extract accession if present in description
        accession = None
        acc_match = ACCESSION_PATTERN.search(description)
        if acc_match:
            accession = acc_match.group(1)
        
        return {
            'species': species_display,
            'species_raw': species,
            'biotype': biotype,
            'description': description,
            'accession': accession,
            'id_type': 'genome_sampler'
        }
    return None

def parse_fasta_file(filepath: str) -> List[Dict[str, Union[str, int]]]:
    """
    Parses a FASTA file (including genome sampler format) and returns sequence data.
    Optimized for large files with streaming.
    """
    sequences = []
    current_header = None
    current_seq_parts = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('>'):
                    # Process previous sequence if exists
                    if current_header and current_seq_parts:
                        seq = ''.join(current_seq_parts)
                        header_data = parse_genome_sampler_header(current_header)
                        if header_data:
                            header_data['sequence'] = seq
                            header_data['length'] = len(seq)
                            sequences.append(header_data)
                    
                    # Start new sequence
                    current_header = line
                    current_seq_parts = []
                else:
                    # Accumulate sequence parts
                    current_seq_parts.append(line)
            
            # Don't forget last sequence
            if current_header and current_seq_parts:
                seq = ''.join(current_seq_parts)
                header_data = parse_genome_sampler_header(current_header)
                if header_data:
                    header_data['sequence'] = seq
                    header_data['length'] = len(seq)
                    sequences.append(header_data)
                    
    except Exception as e:
        logging.error(f"Error parsing FASTA file {filepath}: {e}")
        return []
    
    logging.info(f"Parsed {len(sequences)} sequences from {filepath}")
    return sequences


def process_genome_sampler_target(sampler_data: dict, config_dict: dict = None) -> Optional[dict]:
    """
    Process a target from genome sampler format.
    """
    # Get config values
    min_seq_len = MIN_SEQ_LEN
    max_seq_len = MAX_SEQ_LEN
    if config_dict:
        min_seq_len = config_dict.get('MIN_SEQ_LEN', MIN_SEQ_LEN)
        max_seq_len = config_dict.get('MAX_SEQ_LEN', MAX_SEQ_LEN)
        
    try:
        seq_str = sampler_data.get('sequence', '')
        if not seq_str:
            logging.error(f"No sequence in genome sampler data: {sampler_data}")
            return None
        
        species = sampler_data.get('species', 'Unknown')
        biotype = sampler_data.get('biotype', 'unknown')
        original_id = f"gs_{sampler_data.get('species_raw', 'unknown')}_{biotype}"
        
        # Clean sequence
        cleaned_seq = clean_sequence(seq_str, original_id)
        if cleaned_seq is None:
            return None
        
        # Length validation
        seq_len = len(cleaned_seq)
        if seq_len < min_seq_len:
            logging.warning(f"Genome sampler sequence too short: {seq_len} < {min_seq_len}")
            return None
        
        final_seq = cleaned_seq
        if seq_len > max_seq_len:
            final_seq = cleaned_seq[:max_seq_len]
            seq_len = max_seq_len
        
        # Generate hash ID
        seq_hash = hashlib.sha256(final_seq.encode('ascii')).hexdigest()
        
        return {
            'id': seq_hash,
            'original_id': original_id,
            'species': species,
            'biotype': biotype,
            'sequence': final_seq,
            'length': seq_len
        }
        
    except Exception as e:
        logging.error(f"Error processing genome sampler target: {e}")
        return None

def process_target(target_info: tuple, config_dict: dict = None) -> Optional[dict]:
    """
    Enhanced to handle both original format and genome sampler format.
    """
    # Get config values
    min_seq_len = MIN_SEQ_LEN
    max_seq_len = MAX_SEQ_LEN
    if config_dict:
        min_seq_len = config_dict.get('MIN_SEQ_LEN', MIN_SEQ_LEN)
        max_seq_len = config_dict.get('MAX_SEQ_LEN', MAX_SEQ_LEN)
    
    try:
        # Handle variable-length tuples
        if len(target_info) == 4:
            species, biotype, id_type, identifier = target_info
            label_override = None
        elif len(target_info) == 5:
            species, biotype, id_type, identifier, label_override = target_info
        else:
            # Try to parse as dict (from genome sampler)
            if isinstance(target_info, dict):
                return process_genome_sampler_target(target_info, config_dict)
            else:
                logging.error(f"Invalid target format: {target_info}")
                return None
        
        final_biotype = label_override if label_override else biotype
        id_type_lower = id_type.lower() if isinstance(id_type, str) else ''

        logging.debug(f"--- Processing Target --- Species='{species}', Biotype='{final_biotype}', Type='{id_type}', ID/Seq='{str(identifier)[:50]}...'")

        seq_rec = None          # Holds the Bio.SeqRecord object
        seq_str = None          # Holds the sequence string before cleaning
        original_id_str = f"{id_type}:{str(identifier)[:30]}" # Default identifier for logging

        # --- Step 1: Get the Initial Sequence (Local, Genome Sampler, or Fetched) ---
        if id_type_lower == 'local_sequence':
            if not isinstance(identifier, str) or not identifier:
                logging.error(f"Local sequence target for {species} - {final_biotype} has invalid sequence data. Skipping.")
                return None
            seq_str = identifier
            # Create a simple ID for local sequences
            seq_hash_short = hashlib.sha256(seq_str.encode()).hexdigest()[:8]
            original_id_str = f"local_{safe_filename(species)}_{safe_filename(final_biotype)}_{seq_hash_short}"
            logging.debug(f"Using local sequence. Assigned ID: {original_id_str}, Initial Length: {len(seq_str)}")

        elif id_type_lower == 'genome_sampler':
            # Handle pre-parsed genome sampler data
            if isinstance(identifier, dict) and 'sequence' in identifier:
                seq_str = identifier['sequence']
                original_id_str = f"gs_{identifier.get('species_raw', 'unknown')}_{identifier.get('biotype', 'unknown')}"
                logging.debug(f"Using genome sampler sequence. ID: {original_id_str}, Length: {len(seq_str)}")
            else:
                logging.error(f"Invalid genome sampler data for {species} - {final_biotype}")
                return None

        elif id_type_lower == 'accession':
            if not isinstance(identifier, str) or not identifier:
                 logging.error(f"Accession target for {species} - {final_biotype} has invalid identifier '{identifier}'. Skipping.")
                 return None
            original_id_str = identifier # Use accession as the original ID

            # Determine if GenBank format is needed for feature extraction
            complex_types = ['exon', 'intron', 'trna', 'rrna', 'pseudogene', 'cds',
                             'gene', 'mobile_element', 'repeat_region'] # Lowercase for comparison
            needs_gb = any(ct in final_biotype.lower() for ct in complex_types) or any(ct in biotype.lower() for ct in complex_types)

            fetch_type = NCBI_RETTYPE_GB if needs_gb else NCBI_RETTYPE_FASTA
            logging.debug(f"Fetching accession {identifier} as {fetch_type} format.")

            # Fetch and parse the primary record (FASTA or GenBank)
            record_data = fetch_ncbi_record(identifier, db=NCBI_DB, rettype=fetch_type, retmode=NCBI_RETMODE)
            primary_rec = parse_sequence_data(identifier, record_data, fetch_type)

            if primary_rec is None:
                logging.warning(f"Failed to fetch or parse primary record for accession {identifier} ({fetch_type}). Skipping target.")
                return None
            logging.debug(f"Successfully fetched/parsed primary record {primary_rec.id} (Length: {len(primary_rec.seq)})")
            original_id_str = primary_rec.id # Update original_id with the actual fetched ID

            # --- Step 2: Feature Extraction (if GenBank was required/fetched) ---
            if needs_gb:
                gb_rec = primary_rec # The record we fetched is GenBank
                logging.info(f"Attempting feature extraction for biotype '{final_biotype}' from GenBank record {gb_rec.id}...")
                extracted_rec = None

                # Feature extraction logic
                if 'rrna' in final_biotype.lower():
                    # Example: Extract 16S rRNA if specified
                    if '16s' in final_biotype.lower():
                         qual = {'product': ['16S ribosomal RNA', '16S rRNA']}
                         extracted_rec = get_feature_sequence(gb_rec, 'rRNA', qual, 0)
                    elif '18s' in final_biotype.lower():
                         qual = {'product': ['18S ribosomal RNA', '18S rRNA']}
                         extracted_rec = get_feature_sequence(gb_rec, 'rRNA', qual, 0)
                    else: # Generic rRNA fallback
                         extracted_rec = get_feature_sequence(gb_rec, 'rRNA', None, 0)
                elif 'cds' in final_biotype.lower() or biotype.lower() == 'protein-coding':
                    # Example: Extract the first CDS feature
                    extracted_rec = get_feature_sequence(gb_rec, 'CDS', None, 0)
                elif 'trna' in final_biotype.lower():
                     extracted_rec = get_feature_sequence(gb_rec, 'tRNA', None, 0)
                else:
                    logging.warning(f"No specific extraction logic defined for biotype '{final_biotype}'. Using the full GenBank sequence from {gb_rec.id}.")
                    # Fallback: Use the entire GenBank sequence if no specific feature logic matches
                    extracted_rec = gb_rec

                # --- Check Extraction Result ---
                if extracted_rec is None:
                    logging.error(f"Feature extraction FAILED for biotype '{final_biotype}' from {gb_rec.id}. Skipping target.")
                    return None
                else:
                    logging.debug(f"Using sequence from feature extraction. Extracted Record ID: {extracted_rec.id}, Length: {len(extracted_rec.seq)}")
                    seq_rec = extracted_rec # Use the extracted feature record
                    original_id_str = extracted_rec.id # Update ID to reflect the extracted feature
            else: # needs_gb was False, we fetched FASTA
                seq_rec = primary_rec # Use the FASTA record directly

            # Extract the sequence string from the final SeqRecord
            if seq_rec and hasattr(seq_rec, 'seq') and seq_rec.seq is not None:
                seq_str = str(seq_rec.seq)
            else:
                logging.error(f"Could not obtain sequence string from record {original_id_str} after processing accession. Skipping.")
                return None

        else: # Unknown id_type
            logging.error(f"Unsupported identifier_type '{id_type}' for target: {species} - {final_biotype}. Skipping.")
            return None

        # --- Step 3: Sequence Cleaning ---
        logging.debug(f"Cleaning sequence for {original_id_str} (Length before clean: {len(seq_str)})...")
        cleaned_seq = clean_sequence(seq_str, original_id_str)

        if cleaned_seq is None:
            logging.warning(f"Sequence cleaning failed for {original_id_str}. Skipping target.")
            return None

        # --- Step 4: Length Validation and Trimming ---
        seq_len = len(cleaned_seq)
        logging.debug(f"Cleaned sequence length: {seq_len}")

        if seq_len < min_seq_len:
            logging.debug(f"Sequence '{original_id_str}' is too short after cleaning ({seq_len} bp < {min_seq_len} bp). Skipping.")
            return None

        final_seq_str = cleaned_seq
        if seq_len > max_seq_len:
            logging.debug(f"Sequence '{original_id_str}' is too long ({seq_len} bp > {max_seq_len} bp). Trimming to {max_seq_len} bp.")
            final_seq_str = cleaned_seq[:max_seq_len]
            seq_len = max_seq_len # Update length after trimming

        logging.debug(f"Sequence length validation passed. Final length: {seq_len}")

        # --- Step 5: Final Output ---
        # Use a stable hash of the final sequence as the primary ID for uniqueness
        seq_hash = hashlib.sha256(final_seq_str.encode('ascii')).hexdigest()

        logging.debug(f"Target Processing SUCCESS: Species={species}, Biotype={final_biotype}, OrigID={original_id_str}, FinalLen={seq_len}, Hash={seq_hash[:8]}...")

        return {
            'id': seq_hash,             # Unique ID based on final sequence content
            'original_id': original_id_str, # Identifier before processing
            'species': species,
            'biotype': final_biotype,   # The potentially overridden biotype/label
            'sequence': final_seq_str,  # The final, cleaned, length-validated sequence
            'length': seq_len
        }

    except Exception as e:
        # Catch any unexpected errors during processing of a single target
        logging.error(f"Unexpected error processing target '{target_info}': {type(e).__name__}: {e}", exc_info=True)
        return None

def parse_genome_sampler_header(header_line: str) -> Optional[Dict[str, str]]:
    """
    Parses headers from genome sampler output format.
    Format: >species|biotype|description
    Returns dict with species, biotype, description fields.
    """
    if not header_line or not header_line.startswith('>'):
        return None
    
    # Remove '>' and match pattern
    match = GENOME_SAMPLER_PATTERN.match(header_line)
    if match:
        species, biotype, description = match.groups()
        # Clean up species name (replace underscores with spaces for display)
        species_display = species.replace('_', ' ')
        
        # Fix biotype labeling for random_dna_L<XXX>
        if biotype.startswith('random_dna_L'):
            biotype = 'random_dna'
        
        # Extract accession if present in description
        accession = None
        acc_match = ACCESSION_PATTERN.search(description)
        if acc_match:
            accession = acc_match.group(1)
        
        return {
            'species': species_display,
            'species_raw': species,
            'biotype': biotype,
            'description': description,
            'accession': accession,
            'id_type': 'genome_sampler'
        }
    return None

def process_targets_parallel(targets: List[Union[tuple, dict]], max_workers: int = 4, config_dict: dict = None) -> List[dict]:
    """
    Process multiple targets in parallel for better performance.
    """
    valid_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_target = {executor.submit(process_target, t, config_dict): t for t in targets}
        
        # Collect results as they complete
        for future in as_completed(future_to_target):
            try:
                result = future.result()
                if result is not None:
                    valid_results.append(result)
            except Exception as e:
                target = future_to_target[future]
                logging.error(f"Failed to process target {target}: {e}")
    
    return valid_results

# Helper for safe filename generation
from .utils import safe_filename