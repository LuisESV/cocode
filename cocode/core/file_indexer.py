"""
FileIndexer: Analyze and index the codebase.

This module provides functionality to parse and index a codebase,
build dependency graphs, and provide context for the LLM.
"""

import os
import re
import time
import json
import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

try:
    import tree_sitter
except ImportError:
    tree_sitter = None

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileIndexer:
    """Analyze and index the codebase for improved context."""
    
    # Extensions to ignore during indexing
    IGNORE_EXTENSIONS = {
        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.obj', '.out',
        '.log', '.swp', '.swo', '.env', '.lock', '.git', '.gitkeep',
        '.DS_Store', 'Thumbs.db', '.idea', '.vscode', '.ipynb_checkpoints',
        '.sass-cache', '.pytest_cache', '.coverage', '.mypy_cache',
    }
    
    # Directories to ignore during indexing
    IGNORE_DIRS = {
        '.git', '.svn', '.hg', 'node_modules', 'venv', 'env', '.venv',
        '.env', '__pycache__', '.pytest_cache', '.idea', '.vscode',
        'dist', 'build', '.ipynb_checkpoints', '.mypy_cache',
    }
    
    def __init__(self, work_dir: str = None, max_file_size: int = 1024 * 1024, model_connector=None):
        """
        Initialize the FileIndexer.
        
        Args:
            work_dir: The directory to index (default: current directory)
            max_file_size: Maximum file size to index in bytes (default: 1MB)
            model_connector: ModelConnector instance for embeddings (optional)
        """
        self.work_dir = work_dir or os.getcwd()
        self.max_file_size = max_file_size
        self.model_connector = model_connector
        
        # File data storage
        self.files = {}  # Map of file paths to file data
        self.file_extensions = defaultdict(list)  # Map of extensions to files
        self.file_types = defaultdict(list)  # Map of file types to files
        
        # Symbol data storage
        self.symbols = defaultdict(list)  # Map of symbol names to locations
        self.symbol_types = defaultdict(set)  # Map of symbol names to types (class, function, etc.)
        
        # Dependency graph
        self.imports = defaultdict(set)  # Map of files to their imports
        self.imported_by = defaultdict(set)  # Map of files to files that import them
        
        # Vector embeddings
        self.embeddings = {}  # Map of file paths to embeddings
        self.symbol_embeddings = {}  # Map of symbol IDs to embeddings
        
        # Indexing status
        self.indexed = False
        self.last_indexed = None
        
        # Tree-sitter language parsers
        self.parsers = {}
        if tree_sitter:
            self._setup_parsers()
        
        logger.info(f"FileIndexer initialized for directory: {self.work_dir}")
    
    def _setup_parsers(self):
        """Set up Tree-sitter parsers for supported languages."""
        try:
            # Set Python parser as an example - 
            # In a complete implementation, you'd include 
            # parsers for multiple languages
            PY_LANGUAGE = tree_sitter.Language('build/languages.so', 'python')
            self.parsers['python'] = tree_sitter.Parser()
            self.parsers['python'].set_language(PY_LANGUAGE)
            
            logger.info("Tree-sitter parsers initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Tree-sitter parsers: {e}")
    
    def _should_ignore_path(self, path: str) -> bool:
        """Check if a path should be ignored during indexing."""
        path_parts = path.split(os.sep)
        
        # Check for ignored directories
        for part in path_parts:
            if part in self.IGNORE_DIRS:
                return True
        
        # Check for ignored extensions
        _, ext = os.path.splitext(path)
        if ext.lower() in self.IGNORE_EXTENSIONS:
            return True
        
        return False
    
    def _get_file_type(self, path: str) -> str:
        """Determine the file type based on the file extension."""
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        
        if ext in ['.py']:
            return 'python'
        elif ext in ['.js', '.jsx']:
            return 'javascript'
        elif ext in ['.ts', '.tsx']:
            return 'typescript'
        elif ext in ['.html', '.htm']:
            return 'html'
        elif ext in ['.css', '.scss', '.sass', '.less']:
            return 'css'
        elif ext in ['.json']:
            return 'json'
        elif ext in ['.md']:
            return 'markdown'
        elif ext in ['.yml', '.yaml']:
            return 'yaml'
        elif ext in ['.java']:
            return 'java'
        elif ext in ['.c', '.cpp', '.cc', '.h', '.hpp']:
            return 'c++'
        elif ext in ['.rs']:
            return 'rust'
        elif ext in ['.go']:
            return 'go'
        elif ext in ['.rb']:
            return 'ruby'
        elif ext in ['.php']:
            return 'php'
        else:
            return 'text'
    
    def index(self, force: bool = False, generate_embeddings: bool = True) -> Dict[str, Any]:
        """
        Index the codebase.
        
        Args:
            force: Whether to force reindexing if already indexed
            generate_embeddings: Whether to generate vector embeddings
            
        Returns:
            Dict with indexing statistics
        """
        # Check if already indexed and not forced
        if self.indexed and not force:
            logger.info("Codebase already indexed, skipping")
            return {
                "status": "skipped",
                "last_indexed": self.last_indexed,
                "file_count": len(self.files),
                "message": "Codebase already indexed"
            }
        
        start_time = time.time()
        
        # Reset data structures
        self.files = {}
        self.file_extensions = defaultdict(list)
        self.file_types = defaultdict(list)
        self.symbols = defaultdict(list)
        self.symbol_types = defaultdict(set)
        self.imports = defaultdict(set)
        self.imported_by = defaultdict(set)
        self.embeddings = {}
        self.symbol_embeddings = {}
        
        # Walk the directory tree
        file_count = 0
        skipped_count = 0
        error_count = 0
        
        logger.info(f"Starting indexing of {self.work_dir}")
        
        try:
            for root, dirs, files in os.walk(self.work_dir):
                # Filter out ignored directories in-place
                dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]
                
                # Process each file
                for filename in files:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, self.work_dir)
                    
                    # Skip ignored paths
                    if self._should_ignore_path(rel_path):
                        skipped_count += 1
                        continue
                    
                    # Skip large files
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size > self.max_file_size:
                            logger.info(f"Skipping large file: {rel_path} ({file_size} bytes)")
                            skipped_count += 1
                            continue
                    except Exception as e:
                        logger.warning(f"Error checking file size for {rel_path}: {e}")
                        error_count += 1
                        continue
                    
                    # Process the file
                    try:
                        # Get basic file info
                        file_type = self._get_file_type(filename)
                        _, ext = os.path.splitext(filename)
                        
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Store file data
                        self.files[rel_path] = {
                            'path': rel_path,
                            'type': file_type,
                            'size': file_size,
                            'content': content,  # Store content for embedding
                            'last_modified': os.path.getmtime(file_path),
                            'symbols': [],
                            'imports': [],
                        }
                        
                        # Update indices
                        self.file_extensions[ext.lower()].append(rel_path)
                        self.file_types[file_type].append(rel_path)
                        
                        # Parse imports and symbols
                        self._parse_file(rel_path, content, file_type)
                        
                        file_count += 1
                    except Exception as e:
                        logger.warning(f"Error processing file {rel_path}: {e}")
                        error_count += 1
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to index codebase"
            }
        
        # Build dependency graph (imports/imported_by)
        self._build_dependency_graph()
        
        # Generate embeddings if enabled
        if generate_embeddings and self.model_connector:
            self._generate_embeddings()
        
        # Update indexing status
        self.indexed = True
        self.last_indexed = time.time()
        
        # Calculate statistics
        indexing_time = self.last_indexed - start_time
        
        logger.info(f"Indexing completed: {file_count} files indexed, {skipped_count} skipped, {error_count} errors")
        
        # Remove file content to save memory after embeddings are generated
        for file_data in self.files.values():
            if 'content' in file_data:
                del file_data['content']
        
        return {
            "status": "success",
            "file_count": file_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
            "time_taken": indexing_time,
            "embeddings_count": len(self.embeddings),
            "symbol_embeddings_count": len(self.symbol_embeddings),
            "language_counts": {lang: len(files) for lang, files in self.file_types.items()},
            "message": f"Indexed {file_count} files in {indexing_time:.2f} seconds"
        }
    
    def _build_dependency_graph(self):
        """Build dependency graph from parsed imports."""
        for file_path, imports in self.imports.items():
            for import_stmt in imports:
                # Try to resolve the import to a file path
                # (This is a simplistic implementation)
                for other_path in self.files.keys():
                    # Check if the import might refer to this file
                    # (In a full implementation, this would be more sophisticated)
                    if other_path.endswith(import_stmt.replace('.', '/') + '.py'):
                        self.imported_by[other_path].add(file_path)
    
    def _generate_embeddings(self):
        """Generate vector embeddings for files and symbols."""
        if not self.model_connector:
            logger.warning("No model connector available for embeddings")
            return
        
        logger.info("Generating embeddings for files and symbols")
        
        # Generate file embeddings
        file_batch = []
        file_paths = []
        
        for file_path, file_data in self.files.items():
            if 'content' not in file_data:
                continue
                
            # Prepare a summary of the file for embedding
            content = file_data['content']
            # Truncate long files to avoid exceeding token limits
            if len(content) > 8000:
                content = content[:8000] + "..."
                
            file_batch.append(content)
            file_paths.append(file_path)
            
            # Process in batches to avoid memory issues
            if len(file_batch) >= 10:
                self._process_file_embeddings_batch(file_batch, file_paths)
                file_batch = []
                file_paths = []
        
        # Process any remaining files
        if file_batch:
            self._process_file_embeddings_batch(file_batch, file_paths)
            
        # Generate symbol embeddings
        symbol_batch = []
        symbol_ids = []
        
        for symbol_name, locations in self.symbols.items():
            for location in locations:
                file_path = location['file']
                if file_path not in self.files:
                    continue
                    
                # Create a unique ID for this symbol instance
                symbol_id = f"{file_path}:{location['line']}:{symbol_name}"
                
                # Get file type
                file_type = self.files[file_path]['type']
                
                # Create a context snippet for the symbol
                symbol_context = self._get_symbol_context(file_path, location)
                if not symbol_context:
                    continue
                    
                symbol_batch.append(f"Symbol: {symbol_name}\nType: {location['type']}\nLanguage: {file_type}\nContext:\n{symbol_context}")
                symbol_ids.append(symbol_id)
                
                # Process in batches
                if len(symbol_batch) >= 10:
                    self._process_symbol_embeddings_batch(symbol_batch, symbol_ids)
                    symbol_batch = []
                    symbol_ids = []
        
        # Process any remaining symbols
        if symbol_batch:
            self._process_symbol_embeddings_batch(symbol_batch, symbol_ids)
            
        logger.info(f"Generated embeddings for {len(self.embeddings)} files and {len(self.symbol_embeddings)} symbols")
    
    def _process_file_embeddings_batch(self, file_batch, file_paths):
        """Process a batch of file embeddings."""
        try:
            # Generate embeddings using OpenAI's embeddings API
            response = self._get_embeddings(file_batch)
            
            # Store embeddings
            for i, file_path in enumerate(file_paths):
                if i < len(response):
                    self.embeddings[file_path] = response[i]
        except Exception as e:
            logger.error(f"Error generating file embeddings: {e}")
    
    def _process_symbol_embeddings_batch(self, symbol_batch, symbol_ids):
        """Process a batch of symbol embeddings."""
        try:
            # Generate embeddings using OpenAI's embeddings API
            response = self._get_embeddings(symbol_batch)
            
            # Store embeddings
            for i, symbol_id in enumerate(symbol_ids):
                if i < len(response):
                    self.symbol_embeddings[symbol_id] = response[i]
        except Exception as e:
            logger.error(f"Error generating symbol embeddings: {e}")
    
    def _get_embeddings(self, texts):
        """
        Get embeddings for a list of texts using the model connector.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Make a simple embedding request without streaming
            prompt = "I'll provide code snippets. Generate embeddings that capture the semantic meaning of each snippet. Each snippet is separated by a [SPLIT] marker.\n\n"
            prompt += "[SPLIT]".join(texts)
            
            # Make a request to produce embeddings
            response = self.model_connector.chat(
                user_message=prompt + "\n\nPlease generate embeddings for these code snippets.",
                stream=False
            )
            
            # Extract embeddings from response
            # This is a simplified version - in a real implementation, you'd use
            # OpenAI's embeddings API directly rather than the chat API
            
            # For now, we'll create pseudo-embeddings by hashing the content
            # This is not semantically meaningful but serves as a placeholder
            # until we implement proper embeddings
            embeddings = []
            for text in texts:
                # Create a pseudo-embedding of dimension 128
                hash_val = hash(text)
                np.random.seed(hash_val)
                embedding = np.random.randn(128)
                # Normalize the embedding
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            # Return empty embeddings as fallback
            return [np.zeros(128) for _ in range(len(texts))]
    
    def _get_symbol_context(self, file_path, location):
        """Get the context (surrounding code) for a symbol."""
        if file_path not in self.files or 'content' not in self.files[file_path]:
            return None
            
        content = self.files[file_path]['content']
        lines = content.split('\n')
        line_index = location['line'] - 1  # Convert to 0-based index
        
        if line_index >= len(lines):
            return None
            
        # Get a window of context around the symbol
        start_line = max(0, line_index - 5)
        end_line = min(len(lines), line_index + 10)
        
        return '\n'.join(lines[start_line:end_line])
    
    def _parse_file(self, file_path: str, content: str, file_type: str) -> None:
        """
        Parse a file to extract imports and symbols.
        
        Args:
            file_path: The path to the file
            content: The content of the file
            file_type: The type of the file
        """
        # For Python files, try to use Tree-sitter if available
        if file_type == 'python' and tree_sitter and 'python' in self.parsers:
            self._parse_python_file_tree_sitter(file_path, content)
            return
        
        # Fall back to regex-based parsing for Python
        if file_type == 'python':
            self._parse_python_file_regex(file_path, content)
            return
        
        # For JavaScript/TypeScript files
        if file_type in ['javascript', 'typescript']:
            self._parse_js_ts_file_regex(file_path, content)
            return
        
        # For other file types, use simple regex-based parsing
        # This is a placeholder for a more comprehensive implementation
        self._parse_generic_file_regex(file_path, content)
    
    def _parse_python_file_tree_sitter(self, file_path: str, content: str) -> None:
        """Parse a Python file using Tree-sitter."""
        parser = self.parsers['python']
        tree = parser.parse(bytes(content, 'utf8'))
        root_node = tree.root_node
        
        # Process import statements
        import_nodes = self._query_tree_sitter_nodes(root_node, '(import_statement)')
        for node in import_nodes:
            import_path = content[node.start_byte:node.end_byte].strip()
            if import_path.startswith('import '):
                import_path = import_path[7:]
            
            # Add to imports
            self.files[file_path]['imports'].append(import_path)
            self.imports[file_path].add(import_path)
            
            # Try to resolve the import
            # (In a complete implementation, you'd map the import to a file path)
        
        # Process class definitions
        class_nodes = self._query_tree_sitter_nodes(root_node, '(class_definition)')
        for node in class_nodes:
            name_node = node.child_by_field_name('name')
            if name_node:
                class_name = content[name_node.start_byte:name_node.end_byte]
                
                # Add to symbols
                symbol_data = {
                    'name': class_name,
                    'type': 'class',
                    'file': file_path,
                    'line': name_node.start_point[0] + 1,
                }
                
                self.files[file_path]['symbols'].append(symbol_data)
                self.symbols[class_name].append(symbol_data)
                self.symbol_types[class_name].add('class')
        
        # Process function definitions
        function_nodes = self._query_tree_sitter_nodes(root_node, '(function_definition)')
        for node in function_nodes:
            name_node = node.child_by_field_name('name')
            if name_node:
                function_name = content[name_node.start_byte:name_node.end_byte]
                
                # Add to symbols
                symbol_data = {
                    'name': function_name,
                    'type': 'function',
                    'file': file_path,
                    'line': name_node.start_point[0] + 1,
                }
                
                self.files[file_path]['symbols'].append(symbol_data)
                self.symbols[function_name].append(symbol_data)
                self.symbol_types[function_name].add('function')
    
    def _query_tree_sitter_nodes(self, node, query_str):
        """Helper method to query Tree-sitter nodes."""
        # This is a placeholder for a more complete implementation
        # that would use the tree-sitter query API
        results = []
        
        # Recursive traversal to find nodes matching the query
        def traverse(n):
            if n.type == query_str.strip('()'):
                results.append(n)
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return results
    
    def _parse_python_file_regex(self, file_path: str, content: str) -> None:
        """Parse a Python file using regex patterns."""
        # Extract imports
        import_regex = r'^\s*(import|from)\s+([a-zA-Z0-9_\.]+(?: import [a-zA-Z0-9_\.]+(?:,\s*[a-zA-Z0-9_\.]+)*)?)'
        for match in re.finditer(import_regex, content, re.MULTILINE):
            import_stmt = match.group(0).strip()
            self.files[file_path]['imports'].append(import_stmt)
            self.imports[file_path].add(import_stmt)
        
        # Extract class definitions
        class_regex = r'^\s*class\s+([a-zA-Z0-9_]+)(?:\(.*\))?:'
        for match in re.finditer(class_regex, content, re.MULTILINE):
            class_name = match.group(1)
            
            # Get line number
            line_number = content[:match.start()].count('\n') + 1
            
            # Add to symbols
            symbol_data = {
                'name': class_name,
                'type': 'class',
                'file': file_path,
                'line': line_number,
            }
            
            self.files[file_path]['symbols'].append(symbol_data)
            self.symbols[class_name].append(symbol_data)
            self.symbol_types[class_name].add('class')
        
        # Extract function definitions
        func_regex = r'^\s*def\s+([a-zA-Z0-9_]+)\s*\(.*\):'
        for match in re.finditer(func_regex, content, re.MULTILINE):
            func_name = match.group(1)
            
            # Get line number
            line_number = content[:match.start()].count('\n') + 1
            
            # Add to symbols
            symbol_data = {
                'name': func_name,
                'type': 'function',
                'file': file_path,
                'line': line_number,
            }
            
            self.files[file_path]['symbols'].append(symbol_data)
            self.symbols[func_name].append(symbol_data)
            self.symbol_types[func_name].add('function')
    
    def _parse_js_ts_file_regex(self, file_path: str, content: str) -> None:
        """Parse a JavaScript/TypeScript file using regex patterns."""
        # Extract imports
        import_regex = r'^\s*(import|export|require).*?[\'"]([^\'"]*)[\'"](;?)'
        for match in re.finditer(import_regex, content, re.MULTILINE):
            import_stmt = match.group(0).strip()
            self.files[file_path]['imports'].append(import_stmt)
            self.imports[file_path].add(import_stmt)
        
        # Extract class definitions
        class_regex = r'^\s*(export\s+)?(class)\s+([a-zA-Z0-9_]+)'
        for match in re.finditer(class_regex, content, re.MULTILINE):
            class_name = match.group(3)
            
            # Get line number
            line_number = content[:match.start()].count('\n') + 1
            
            # Add to symbols
            symbol_data = {
                'name': class_name,
                'type': 'class',
                'file': file_path,
                'line': line_number,
            }
            
            self.files[file_path]['symbols'].append(symbol_data)
            self.symbols[class_name].append(symbol_data)
            self.symbol_types[class_name].add('class')
        
        # Extract function definitions
        func_regex = r'^\s*(export\s+)?(function|const|let|var)\s+([a-zA-Z0-9_]+)\s*(?:=\s*(?:function|\([^)]*\)\s*=>)|[:(])'
        for match in re.finditer(func_regex, content, re.MULTILINE):
            func_name = match.group(3)
            
            # Get line number
            line_number = content[:match.start()].count('\n') + 1
            
            # Add to symbols
            symbol_data = {
                'name': func_name,
                'type': 'function',
                'file': file_path,
                'line': line_number,
            }
            
            self.files[file_path]['symbols'].append(symbol_data)
            self.symbols[func_name].append(symbol_data)
            self.symbol_types[func_name].add('function')
    
    def _parse_generic_file_regex(self, file_path: str, content: str) -> None:
        """Parse a generic file using simple regex patterns."""
        # This is a minimal implementation that just looks for 
        # common patterns like functions or classes
        
        # Simple pattern to catch function-like definitions in various languages
        func_regex = r'(?:function|def|fn|func)\s+([a-zA-Z0-9_]+)\s*\('
        for match in re.finditer(func_regex, content):
            func_name = match.group(1)
            
            # Get line number
            line_number = content[:match.start()].count('\n') + 1
            
            # Add to symbols
            symbol_data = {
                'name': func_name,
                'type': 'function',
                'file': file_path,
                'line': line_number,
            }
            
            self.files[file_path]['symbols'].append(symbol_data)
            self.symbols[func_name].append(symbol_data)
            self.symbol_types[func_name].add('function')
        
        # Simple pattern to catch class-like definitions in various languages
        class_regex = r'(?:class|struct|interface)\s+([a-zA-Z0-9_]+)'
        for match in re.finditer(class_regex, content):
            class_name = match.group(1)
            
            # Get line number
            line_number = content[:match.start()].count('\n') + 1
            
            # Add to symbols
            symbol_data = {
                'name': class_name,
                'type': 'class',
                'file': file_path,
                'line': line_number,
            }
            
            self.files[file_path]['symbols'].append(symbol_data)
            self.symbols[class_name].append(symbol_data)
            self.symbol_types[class_name].add('class')
    
    def search_files(self, query: str, file_type: str = None, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search for files matching a query.
        
        Args:
            query: The search query
            file_type: Optional file type filter
            max_results: Maximum number of results to return
            
        Returns:
            List of matching files
        """
        if not self.indexed:
            logger.warning("Codebase not indexed, indexing now")
            self.index()
        
        results = []
        query = query.lower()
        
        # Search in all files or filter by type
        files_to_search = self.files
        if file_type:
            files_to_search = {
                path: data for path, data in self.files.items()
                if data['type'] == file_type
            }
        
        # Try semantic search if embeddings are available
        if self.embeddings and self.model_connector:
            semantic_results = self.semantic_search_files(query, max_results=max_results)
            if semantic_results:
                # Combine with exact match results
                results.extend(semantic_results)
        
        # Search for exact matches
        for path, data in files_to_search.items():
            # Check if query is in path
            if query in path.lower():
                # Check if this path is already in results
                if not any(r['path'] == path for r in results):
                    results.append({
                        'path': path,
                        'type': data['type'],
                        'match_type': 'path',
                        'relevance': 0.9  # High relevance for path matches
                    })
                    continue
            
            # Check file symbols
            for symbol in data['symbols']:
                if query in symbol['name'].lower():
                    # Check if this path is already in results
                    if not any(r['path'] == path for r in results):
                        results.append({
                            'path': path,
                            'type': data['type'],
                            'match_type': f"symbol:{symbol['type']}",
                            'symbol': symbol['name'],
                            'line': symbol['line'],
                            'relevance': 0.8  # High relevance for symbol matches
                        })
                        break
        
        # Sort by relevance
        results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        
        # Limit results
        return results[:max_results]
    
    def semantic_search_files(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search on files using embeddings.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of matching files with relevance scores
        """
        if not self.embeddings or not self.model_connector:
            return []
        
        # Get embedding for the query
        query_embedding = self._get_embeddings([query])[0]
        
        # Calculate similarity with all file embeddings
        similarities = []
        for file_path, embedding in self.embeddings.items():
            # Calculate cosine similarity
            similarity = float(cosine_similarity([query_embedding], [embedding])[0][0])
            similarities.append((file_path, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to result format
        results = []
        for file_path, similarity in similarities[:max_results]:
            if file_path in self.files:
                results.append({
                    'path': file_path,
                    'type': self.files[file_path]['type'],
                    'match_type': 'semantic',
                    'relevance': similarity
                })
        
        return results
    
    def search_symbols(self, query: str, symbol_type: str = None, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search for symbols matching a query.
        
        Args:
            query: The search query
            symbol_type: Optional symbol type filter (class, function, etc.)
            max_results: Maximum number of results to return
            
        Returns:
            List of matching symbols
        """
        if not self.indexed:
            logger.warning("Codebase not indexed, indexing now")
            self.index()
        
        results = []
        query = query.lower()
        
        # Try semantic search if embeddings are available
        if self.symbol_embeddings and self.model_connector:
            semantic_results = self.semantic_search_symbols(query, symbol_type, max_results=max_results)
            if semantic_results:
                # Combine with exact match results
                results.extend(semantic_results)
        
        # Search in all symbols
        for symbol_name, locations in self.symbols.items():
            # Check if query is in symbol name
            if query in symbol_name.lower():
                # Check type if specified
                if symbol_type and symbol_type not in self.symbol_types.get(symbol_name, set()):
                    continue
                
                # Add each location as a result
                for location in locations:
                    # Create a unique ID for this symbol instance
                    symbol_id = f"{location['file']}:{location['line']}:{symbol_name}"
                    
                    # Check if this symbol is already in results
                    if not any(r.get('symbol_id') == symbol_id for r in results):
                        results.append({
                            'symbol': symbol_name,
                            'symbol_id': symbol_id,
                            'type': location['type'],
                            'file': location['file'],
                            'line': location['line'],
                            'match_type': 'exact',
                            'relevance': 0.9  # High relevance for exact symbol matches
                        })
        
        # Sort by relevance
        results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        
        # Limit results
        return results[:max_results]
    
    def semantic_search_symbols(self, query: str, symbol_type: str = None, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search on symbols using embeddings.
        
        Args:
            query: The search query
            symbol_type: Optional symbol type filter
            max_results: Maximum number of results to return
            
        Returns:
            List of matching symbols with relevance scores
        """
        if not self.symbol_embeddings or not self.model_connector:
            return []
        
        # Get embedding for the query
        query_embedding = self._get_embeddings([query])[0]
        
        # Calculate similarity with all symbol embeddings
        similarities = []
        for symbol_id, embedding in self.symbol_embeddings.items():
            # Parse symbol ID
            parts = symbol_id.split(':', 2)
            if len(parts) != 3:
                continue
                
            file_path, line_str, symbol_name = parts
            
            # Check symbol type if specified
            if symbol_type and symbol_type not in self.symbol_types.get(symbol_name, set()):
                continue
                
            # Calculate cosine similarity
            similarity = float(cosine_similarity([query_embedding], [embedding])[0][0])
            similarities.append((symbol_id, symbol_name, file_path, line_str, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[4], reverse=True)
        
        # Convert to result format
        results = []
        for symbol_id, symbol_name, file_path, line_str, similarity in similarities[:max_results]:
            # Determine symbol type
            symbol_type = list(self.symbol_types.get(symbol_name, ['unknown']))[0]
            
            results.append({
                'symbol': symbol_name,
                'symbol_id': symbol_id,
                'type': symbol_type,
                'file': file_path,
                'line': int(line_str),
                'match_type': 'semantic',
                'relevance': similarity
            })
        
        return results
    
    def get_file_summary(self, file_path: str) -> Dict[str, Any]:
        """
        Get a summary of a file.
        
        Args:
            file_path: The path to the file
            
        Returns:
            Dict with file summary information
        """
        if not self.indexed:
            logger.warning("Codebase not indexed, indexing now")
            self.index()
        
        # Convert to relative path if needed
        if file_path.startswith(self.work_dir):
            file_path = os.path.relpath(file_path, self.work_dir)
        
        # Check if file is indexed
        if file_path not in self.files:
            return {
                "error": f"File not found in index: {file_path}"
            }
        
        file_data = self.files[file_path]
        
        # Get imports
        imports = list(self.imports.get(file_path, set()))
        
        # Get imported by
        imported_by = list(self.imported_by.get(file_path, set()))
        
        # Get symbols
        symbols = file_data['symbols']
        
        # Check if file has embedding
        has_embedding = file_path in self.embeddings
        
        # Build summary
        summary = {
            "path": file_path,
            "type": file_data['type'],
            "size": file_data['size'],
            "last_modified": file_data['last_modified'],
            "symbols": symbols,
            "imports": imports,
            "imported_by": imported_by,
            "has_embedding": has_embedding,
        }
        
        return summary
    
    def find_similar_files(self, file_path: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find files similar to the given file using vector embeddings.
        
        Args:
            file_path: The path to the file to find similar files for
            max_results: Maximum number of results to return
            
        Returns:
            List of similar files with similarity scores
        """
        if not self.indexed:
            logger.warning("Codebase not indexed, indexing now")
            self.index()
        
        # Convert to relative path if needed
        if file_path.startswith(self.work_dir):
            file_path = os.path.relpath(file_path, self.work_dir)
        
        # Check if file is indexed and has embedding
        if file_path not in self.files or file_path not in self.embeddings:
            return [{
                "error": f"File not found in index or has no embedding: {file_path}"
            }]
        
        # Get the file's embedding
        file_embedding = self.embeddings[file_path]
        
        # Calculate similarity with all other file embeddings
        similarities = []
        for other_path, embedding in self.embeddings.items():
            # Skip the file itself
            if other_path == file_path:
                continue
                
            # Calculate cosine similarity
            similarity = float(cosine_similarity([file_embedding], [embedding])[0][0])
            similarities.append((other_path, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to result format
        results = []
        for other_path, similarity in similarities[:max_results]:
            if other_path in self.files:
                results.append({
                    'path': other_path,
                    'type': self.files[other_path]['type'],
                    'similarity': similarity
                })
        
        return results
    
    def get_context_for_query(self, query: str, max_files: int = 5) -> str:
        """
        Get context information for a query to enhance LLM responses.
        
        Args:
            query: The user's query
            max_files: Maximum number of files to include in context
            
        Returns:
            A string with relevant context for the query
        """
        # Check if indexing is needed
        if not self.indexed:
            logger.warning("Codebase not indexed, indexing now")
            self.index()
        
        context_parts = []
        
        # Use semantic search if available
        if self.embeddings and self.model_connector:
            search_results = self.semantic_search_files(query, max_results=max_files)
            
            if search_results:
                context_parts.append(f"Files semantically relevant to '{query}':")
                for result in search_results:
                    summary = self.get_file_summary(result['path'])
                    
                    # Add file path and type with similarity score
                    context_parts.append(f"- {result['path']} ({summary['type']}, similarity: {result['relevance']:.2f})")
                    
                    # Add top symbols if available
                    if summary.get('symbols'):
                        for symbol in summary['symbols'][:3]:  # Limit to top 3 symbols
                            context_parts.append(f"  - {symbol['type']}: {symbol['name']}")
                            
                return "\n".join(context_parts)
        
        # Define keyword patterns for different types of queries
        file_patterns = [
            r'(file|open|edit|read|create)\s+([a-zA-Z0-9_\./\\-]+\.[a-zA-Z0-9]+)',
            r'(show|display|content of)\s+([a-zA-Z0-9_\./\\-]+\.[a-zA-Z0-9]+)',
        ]
        
        symbol_patterns = [
            r'(function|class|method|def)\s+([a-zA-Z0-9_]+)',
            r'(how|what)\s+(does|is)\s+([a-zA-Z0-9_]+)',
        ]
        
        # Check for file references
        for pattern in file_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                file_name = match.group(2)
                
                # Search for the file
                matches = self.search_files(file_name, max_results=max_files)
                
                if matches:
                    context_parts.append(f"Files matching '{file_name}':")
                    for match in matches:
                        summary = self.get_file_summary(match['path'])
                        
                        # Add file path and type
                        context_parts.append(f"- {match['path']} ({summary['type']})")
                        
                        # Add top symbols if available
                        if summary.get('symbols'):
                            for symbol in summary['symbols'][:3]:  # Limit to top 3 symbols
                                context_parts.append(f"  - {symbol['type']}: {symbol['name']}")
        
        # Check for symbol references
        for pattern in symbol_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                symbol_name = match.groups()[-1]  # Last group is the symbol name
                
                # Search for the symbol
                matches = self.search_symbols(symbol_name, max_results=max_files)
                
                if matches:
                    context_parts.append(f"Symbols matching '{symbol_name}':")
                    for match in matches:
                        context_parts.append(f"- {match['symbol']} ({match['type']}) in {match['file']}:{match['line']}")
        
        # If no specific context was found but the query seems to be about the codebase structure
        if not context_parts and any(keyword in query.lower() for keyword in [
            'structure', 'codebase', 'repo', 'repository', 'project', 'directory', 'file'
        ]):
            # Provide a summary of the codebase structure
            context_parts.append("Codebase structure summary:")
            
            # Count files by type
            type_counts = defaultdict(int)
            for data in self.files.values():
                type_counts[data['type']] += 1
            
            context_parts.append("File types in the codebase:")
            for file_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                context_parts.append(f"- {file_type}: {count} files")
            
            # Add top-level directories
            top_dirs = set()
            for path in self.files.keys():
                parts = path.split('/')
                if len(parts) > 1:
                    top_dirs.add(parts[0])
            
            if top_dirs:
                context_parts.append("\nTop-level directories:")
                for dir_name in sorted(top_dirs):
                    context_parts.append(f"- {dir_name}/")
        
        # Combine all context parts
        if context_parts:
            return "\n".join(context_parts)
        else:
            return ""  # No relevant context found
    
    def get_dependencies(self, file_path: str) -> Dict[str, Any]:
        """
        Get dependencies for a file.
        
        Args:
            file_path: The path to the file
            
        Returns:
            Dict with dependency information
        """
        if not self.indexed:
            logger.warning("Codebase not indexed, indexing now")
            self.index()
        
        # Convert to relative path if needed
        if file_path.startswith(self.work_dir):
            file_path = os.path.relpath(file_path, self.work_dir)
        
        # Check if file is indexed
        if file_path not in self.files:
            return {
                "error": f"File not found in index: {file_path}"
            }
        
        # Get imports (dependencies)
        imports = list(self.imports.get(file_path, set()))
        
        # Get imported by (dependents)
        imported_by = list(self.imported_by.get(file_path, set()))
        
        return {
            "path": file_path,
            "imports": imports,
            "imported_by": imported_by
        }