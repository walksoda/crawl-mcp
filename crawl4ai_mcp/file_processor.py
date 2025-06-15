"""
File Processing Module for MarkItDown Integration
Handles conversion of PDF, Office, and ZIP files to Markdown format
"""

import asyncio
import io
import os
import tempfile
import zipfile
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests
from markitdown import MarkItDown
import base64
from urllib.parse import urlparse, unquote
import logging

class FileProcessor:
    """Process various file formats using MarkItDown"""
    
    def __init__(self):
        self.markitdown = MarkItDown()
        self.supported_extensions = {
            # PDF files
            '.pdf': 'PDF Document',
            # Microsoft Office files
            '.docx': 'Microsoft Word Document',
            '.pptx': 'Microsoft PowerPoint Presentation', 
            '.xlsx': 'Microsoft Excel Spreadsheet',
            '.xls': 'Microsoft Excel Spreadsheet (Legacy)',
            # Archive files
            '.zip': 'ZIP Archive',
            # Other supported formats
            '.html': 'HTML Document',
            '.htm': 'HTML Document',
            '.txt': 'Text File',
            '.md': 'Markdown File',
            '.csv': 'CSV File',
            '.rtf': 'Rich Text Format',
            '.epub': 'EPUB eBook'
        }
    
    def is_supported_file(self, file_path_or_url: str) -> bool:
        """Check if file format is supported"""
        try:
            # Extract file extension
            if file_path_or_url.startswith('http'):
                parsed = urlparse(file_path_or_url)
                path = unquote(parsed.path)
            else:
                path = file_path_or_url
            
            ext = Path(path).suffix.lower()
            is_supported = ext in self.supported_extensions
            
            # If no extension found but it might be a known format, try to infer
            if not is_supported and file_path_or_url.startswith('http'):
                # Check for common patterns
                if '/html' in file_path_or_url or 'html' in file_path_or_url:
                    return True
                elif 'README' in file_path_or_url.upper() and not ext:
                    return True
            
            return is_supported
        except Exception:
            return False
    
    def get_file_type(self, file_path_or_url: str) -> Optional[str]:
        """Get human-readable file type description"""
        try:
            if file_path_or_url.startswith('http'):
                parsed = urlparse(file_path_or_url)
                path = unquote(parsed.path)
            else:
                path = file_path_or_url
            
            ext = Path(path).suffix.lower()
            file_type = self.supported_extensions.get(ext)
            
            # If no extension found but it might be a known format, try to infer
            if not file_type and file_path_or_url.startswith('http'):
                # Check for common patterns
                if '/html' in file_path_or_url or 'html' in file_path_or_url:
                    return 'HTML Document'
                elif 'README' in file_path_or_url.upper() and not ext:
                    return 'Text File'
            
            return file_type
        except Exception:
            return None
    
    async def download_file(self, url: str, max_size_mb: int = 100) -> bytes:
        """Download file from URL with size limit"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > max_size_mb:
                    raise ValueError(f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)")
            
            # Download with size limit
            content = b""
            max_bytes = max_size_mb * 1024 * 1024
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content += chunk
                    if len(content) > max_bytes:
                        raise ValueError(f"File too large: exceeds {max_size_mb}MB limit")
            
            return content
            
        except requests.RequestException as e:
            raise ValueError(f"Failed to download file: {str(e)}")
    
    def extract_zip_contents(self, zip_data: bytes) -> Dict[str, Any]:
        """Extract and process contents of ZIP file"""
        extracted_files = []
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                for file_name in file_list:
                    # Skip directories and hidden files
                    if file_name.endswith('/') or file_name.startswith('.'):
                        continue
                    
                    try:
                        # Check if file is supported
                        if not self.is_supported_file(file_name):
                            extracted_files.append({
                                'name': file_name,
                                'type': 'unsupported',
                                'size': zip_ref.getinfo(file_name).file_size,
                                'content': None,
                                'error': 'Unsupported file format'
                            })
                            continue
                        
                        # Extract file content
                        with zip_ref.open(file_name) as file:
                            file_content = file.read()
                            
                            # Create temporary file for MarkItDown processing
                            with tempfile.NamedTemporaryFile(suffix=Path(file_name).suffix, delete=False) as temp_file:
                                temp_file.write(file_content)
                                temp_file.flush()
                                
                                try:
                                    # Convert to markdown
                                    result = self.markitdown.convert(temp_file.name)
                                    extracted_files.append({
                                        'name': file_name,
                                        'type': self.get_file_type(file_name),
                                        'size': len(file_content),
                                        'content': result.text_content,
                                        'title': getattr(result, 'title', None),
                                        'metadata': getattr(result, 'metadata', {}),
                                        'error': None
                                    })
                                except Exception as e:
                                    extracted_files.append({
                                        'name': file_name,
                                        'type': self.get_file_type(file_name),
                                        'size': len(file_content),
                                        'content': None,
                                        'error': f"Conversion failed: {str(e)}"
                                    })
                                finally:
                                    # Clean up temp file
                                    try:
                                        os.unlink(temp_file.name)
                                    except:
                                        pass
                    
                    except Exception as e:
                        extracted_files.append({
                            'name': file_name,
                            'type': 'error',
                            'size': 0,
                            'content': None,
                            'error': f"Extraction failed: {str(e)}"
                        })
                
                return {
                    'total_files': len(file_list),
                    'processed_files': len(extracted_files),
                    'files': extracted_files
                }
        
        except zipfile.BadZipFile:
            raise ValueError("Invalid ZIP file format")
        except Exception as e:
            raise ValueError(f"ZIP processing failed: {str(e)}")
    
    async def process_file_from_url(self, url: str, max_size_mb: int = 100) -> Dict[str, Any]:
        """Process file from URL"""
        # Get file type early to avoid reference errors
        file_type = self.get_file_type(url)
        
        if not self.is_supported_file(url):
            return {
                'success': False,
                'error': f"Unsupported file format. Supported: {', '.join(self.supported_extensions.keys())}",
                'file_type': file_type
            }
        
        try:
            # Download file
            file_data = await self.download_file(url, max_size_mb)
            
            # Handle ZIP files specially
            if url.lower().endswith('.zip'):
                zip_contents = self.extract_zip_contents(file_data)
                return {
                    'success': True,
                    'url': url,
                    'file_type': file_type,
                    'size_bytes': len(file_data),
                    'is_archive': True,
                    'content': None,
                    'archive_contents': zip_contents
                }
            
            # Process single file
            with tempfile.NamedTemporaryFile(suffix=Path(url).suffix, delete=False) as temp_file:
                temp_file.write(file_data)
                temp_file.flush()
                
                try:
                    result = self.markitdown.convert(temp_file.name)
                    return {
                        'success': True,
                        'url': url,
                        'file_type': file_type,
                        'size_bytes': len(file_data),
                        'is_archive': False,
                        'content': result.text_content,
                        'title': getattr(result, 'title', None),
                        'metadata': getattr(result, 'metadata', {})
                    }
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url,
                'file_type': file_type
            }
    
    async def process_file_from_data(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Process file from binary data"""
        # Get file type early to avoid reference errors
        file_type = self.get_file_type(filename)
        
        if not self.is_supported_file(filename):
            return {
                'success': False,
                'error': f"Unsupported file format. Supported: {', '.join(self.supported_extensions.keys())}",
                'file_type': file_type
            }
        
        try:
            
            # Handle ZIP files specially
            if filename.lower().endswith('.zip'):
                zip_contents = self.extract_zip_contents(file_data)
                return {
                    'success': True,
                    'filename': filename,
                    'file_type': file_type,
                    'size_bytes': len(file_data),
                    'is_archive': True,
                    'content': None,
                    'archive_contents': zip_contents
                }
            
            # Process single file
            with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as temp_file:
                temp_file.write(file_data)
                temp_file.flush()
                
                try:
                    result = self.markitdown.convert(temp_file.name)
                    return {
                        'success': True,
                        'filename': filename,
                        'file_type': file_type,
                        'size_bytes': len(file_data),
                        'is_archive': False,
                        'content': result.text_content,
                        'title': getattr(result, 'title', None),
                        'metadata': getattr(result, 'metadata', {})
                    }
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filename': filename,
                'file_type': file_type
            }