"""
GitAgent: Interface for Git operations.

This module provides a high-level interface for Git operations,
with integration with the LLM for commit message generation.
"""

import os
import re
import logging
import subprocess
from typing import List, Dict, Any, Optional, Tuple, Union

import git
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitAgent:
    """Git integration for CoCoDe."""
    
    def __init__(self, model_connector=None, work_dir: str = None):
        """
        Initialize the GitAgent.
        
        Args:
            model_connector: The ModelConnector for LLM integration
            work_dir: The working directory (defaults to current directory)
        """
        self.model_connector = model_connector
        self.work_dir = work_dir or os.getcwd()
        self.repo = None
        
        # Try to initialize the repo
        try:
            self.repo = Repo(self.work_dir)
            logger.info(f"Git repository found in {self.work_dir}")
        except InvalidGitRepositoryError:
            logger.info(f"No Git repository found in {self.work_dir}")
            self.repo = None
        
        # Set confirmation callback
        self.confirm_callback = None
    
    def set_confirm_callback(self, callback):
        """Set the callback function for confirming Git operations."""
        self.confirm_callback = callback
    
    def is_git_repo(self) -> bool:
        """Check if the current directory is a Git repository."""
        return self.repo is not None
    
    def init_repo(self) -> bool:
        """Initialize a new Git repository in the current directory."""
        if self.is_git_repo():
            logger.info(f"Git repository already exists in {self.work_dir}")
            return True
        
        try:
            self.repo = Repo.init(self.work_dir)
            logger.info(f"Initialized new Git repository in {self.work_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Git repository: {e}")
            return False
    
    def get_status(self) -> Dict[str, List[str]]:
        """Get the status of the Git repository."""
        if not self.is_git_repo():
            return {"error": ["Not a Git repository"]}
        
        try:
            status = {
                "untracked": [],
                "modified": [],
                "deleted": [],
                "staged": [],
                "branches": [],
                "current_branch": None,
            }
            
            # Get the current branch
            status["current_branch"] = self.repo.active_branch.name
            
            # Get all branches
            for branch in self.repo.branches:
                status["branches"].append(branch.name)
            
            # Get file status
            for file in self.repo.untracked_files:
                status["untracked"].append(file)
            
            for file in self.repo.index.diff(None):
                if file.deleted_file:
                    status["deleted"].append(file.a_path)
                else:
                    status["modified"].append(file.a_path)
            
            for file in self.repo.index.diff("HEAD"):
                status["staged"].append(file.a_path)
            
            return status
        except Exception as e:
            logger.error(f"Failed to get Git status: {e}")
            return {"error": [str(e)]}
    
    def get_diff(self, file_path: str = None, staged: bool = False) -> str:
        """
        Get the diff for a file or the entire repository.
        
        Args:
            file_path: The path to the file to get the diff for (optional)
            staged: Whether to get the diff for staged changes (optional)
            
        Returns:
            The diff as a string
        """
        if not self.is_git_repo():
            return "Error: Not a Git repository"
        
        try:
            args = ["git", "diff", "--color=never"]
            if staged:
                args.append("--staged")
            if file_path:
                args.append("--")
                args.append(file_path)
            
            result = subprocess.run(
                args,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            return result.stdout or "No changes"
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get diff: {e}")
            return f"Error: {e.stderr}"
        except Exception as e:
            logger.error(f"Failed to get diff: {e}")
            return f"Error: {str(e)}"
    
    def stage_files(self, paths: List[str]) -> Dict[str, Any]:
        """
        Stage files for commit.
        
        Args:
            paths: List of file paths to stage
            
        Returns:
            Dict with status of the operation
        """
        if not self.is_git_repo():
            return {"success": False, "error": "Not a Git repository"}
        
        try:
            # Convert to list if a single string was passed
            if isinstance(paths, str):
                paths = [paths]
            
            # Get the status before staging
            before_status = self.get_status()
            
            # Stage the files
            self.repo.git.add(paths)
            
            # Get the status after staging
            after_status = self.get_status()
            
            # Compute the changes
            staged_files = []
            for path in paths:
                if path in before_status["untracked"] and path in after_status["staged"]:
                    staged_files.append(f"Added: {path}")
                elif path in before_status["modified"] and path in after_status["staged"]:
                    staged_files.append(f"Modified: {path}")
                elif path in before_status["deleted"] and path in after_status["staged"]:
                    staged_files.append(f"Deleted: {path}")
            
            return {
                "success": True,
                "staged_files": staged_files,
                "message": f"Staged {len(staged_files)} files"
            }
        except Exception as e:
            logger.error(f"Failed to stage files: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def stage_all(self) -> Dict[str, Any]:
        """
        Stage all changed files.
        
        Returns:
            Dict with status of the operation
        """
        if not self.is_git_repo():
            return {"success": False, "error": "Not a Git repository"}
        
        try:
            # Get the status before staging
            before_status = self.get_status()
            
            # Stage all changes
            self.repo.git.add("--all")
            
            # Get the status after staging
            after_status = self.get_status()
            
            # Count the changes
            total_staged = len(after_status["staged"])
            
            return {
                "success": True,
                "message": f"Staged all changes ({total_staged} files)"
            }
        except Exception as e:
            logger.error(f"Failed to stage all files: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def unstage_files(self, paths: List[str]) -> Dict[str, Any]:
        """
        Unstage files.
        
        Args:
            paths: List of file paths to unstage
            
        Returns:
            Dict with status of the operation
        """
        if not self.is_git_repo():
            return {"success": False, "error": "Not a Git repository"}
        
        try:
            # Convert to list if a single string was passed
            if isinstance(paths, str):
                paths = [paths]
            
            # Unstage the files
            self.repo.git.restore("--staged", paths)
            
            return {
                "success": True,
                "message": f"Unstaged {len(paths)} files"
            }
        except Exception as e:
            logger.error(f"Failed to unstage files: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_commit_message(self, diff: str = None) -> str:
        """
        Generate a commit message using the LLM.
        
        Args:
            diff: The diff to generate a commit message for (optional)
            
        Returns:
            The generated commit message
        """
        if not self.model_connector:
            return "Unable to generate commit message: no model connector available"
        
        if not diff:
            diff = self.get_diff(staged=True)
        
        if not diff or diff == "No changes":
            return "No changes to commit"
        
        # Limit the diff size to avoid overwhelming the model
        if len(diff) > 8000:
            diff = diff[:8000] + f"\n\n... [Diff truncated, total length: {len(diff)} characters]"
        
        # Prompt the model to generate a commit message
        prompt = f"""Generate a Git commit message for the following changes.

Guidelines:
- Follow Angular commit message format: <type>(<scope>): <subject>
- Types: feat, fix, docs, style, refactor, test, chore
- Keep the subject line under 72 characters
- Use the imperative mood (e.g., "add" not "added" or "adds")
- No period at the end of the subject line
- Provide a brief explanation of the changes

Diff:
{diff}

Commit message (Angular format):"""
        
        try:
            # Use a non-streaming call for simplicity
            response = self.model_connector.chat(
                user_message=prompt,
                stream=False
            )
            
            # Extract the commit message from the response
            commit_message = response.choices[0].message.content.strip()
            
            # Clean up the message if needed (remove quotes, etc.)
            commit_message = re.sub(r'^["\']|["\']$', '', commit_message)
            
            return commit_message
        except Exception as e:
            logger.error(f"Failed to generate commit message: {e}")
            return "Unable to generate commit message"
    
    def commit(self, message: str = None, generate_message: bool = False) -> Dict[str, Any]:
        """
        Commit staged changes.
        
        Args:
            message: The commit message (optional)
            generate_message: Whether to generate a commit message using the LLM (optional)
            
        Returns:
            Dict with status of the operation
        """
        if not self.is_git_repo():
            return {"success": False, "error": "Not a Git repository"}
        
        try:
            # Check if there are staged changes
            diff = self.get_diff(staged=True)
            if not diff or diff == "No changes":
                return {
                    "success": False,
                    "error": "No changes staged for commit"
                }
            
            # Generate or use provided commit message
            if generate_message:
                message = self.generate_commit_message(diff)
            
            if not message:
                return {
                    "success": False,
                    "error": "No commit message provided"
                }
            
            # Ask for confirmation if callback is set
            if self.confirm_callback:
                confirmed = self.confirm_callback(
                    f"Commit with message: {message}",
                    diff
                )
                if not confirmed:
                    return {
                        "success": False,
                        "message": "Commit cancelled by user"
                    }
            
            # Commit the changes
            commit = self.repo.index.commit(message)
            
            return {
                "success": True,
                "commit_hash": commit.hexsha,
                "message": f"Committed changes with message: {message}"
            }
        except Exception as e:
            logger.error(f"Failed to commit changes: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def push(self, remote: str = "origin", branch: str = None) -> Dict[str, Any]:
        """
        Push commits to a remote repository.
        
        Args:
            remote: The remote to push to (default: "origin")
            branch: The branch to push (default: current branch)
            
        Returns:
            Dict with status of the operation
        """
        if not self.is_git_repo():
            return {"success": False, "error": "Not a Git repository"}
        
        try:
            # Get the current branch if not specified
            if not branch:
                branch = self.repo.active_branch.name
            
            # Ask for confirmation if callback is set
            if self.confirm_callback:
                confirmed = self.confirm_callback(
                    f"Push to {remote}/{branch}?",
                    f"This will push all commits to {remote}/{branch}."
                )
                if not confirmed:
                    return {
                        "success": False,
                        "message": "Push cancelled by user"
                    }
            
            # Push the changes
            push_info = self.repo.remote(remote).push(refspec=f"{branch}:{branch}")
            
            # Check push results
            for info in push_info:
                if info.flags & info.ERROR:
                    return {
                        "success": False,
                        "error": f"Push failed: {info.summary}"
                    }
            
            return {
                "success": True,
                "message": f"Pushed to {remote}/{branch}"
            }
        except Exception as e:
            logger.error(f"Failed to push changes: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_branch(self, branch_name: str) -> Dict[str, Any]:
        """
        Create a new branch.
        
        Args:
            branch_name: The name of the branch to create
            
        Returns:
            Dict with status of the operation
        """
        if not self.is_git_repo():
            return {"success": False, "error": "Not a Git repository"}
        
        try:
            # Check if branch already exists
            if branch_name in [b.name for b in self.repo.branches]:
                return {
                    "success": False,
                    "error": f"Branch '{branch_name}' already exists"
                }
            
            # Create the branch
            self.repo.create_head(branch_name)
            
            return {
                "success": True,
                "message": f"Created branch: {branch_name}"
            }
        except Exception as e:
            logger.error(f"Failed to create branch: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def checkout_branch(self, branch_name: str) -> Dict[str, Any]:
        """
        Checkout a branch.
        
        Args:
            branch_name: The name of the branch to checkout
            
        Returns:
            Dict with status of the operation
        """
        if not self.is_git_repo():
            return {"success": False, "error": "Not a Git repository"}
        
        try:
            # Check if branch exists
            if branch_name not in [b.name for b in self.repo.branches]:
                return {
                    "success": False,
                    "error": f"Branch '{branch_name}' does not exist"
                }
            
            # Checkout the branch
            self.repo.git.checkout(branch_name)
            
            return {
                "success": True,
                "message": f"Checked out branch: {branch_name}"
            }
        except Exception as e:
            logger.error(f"Failed to checkout branch: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_commit_history(self, max_count: int = 10) -> List[Dict[str, str]]:
        """
        Get the commit history.
        
        Args:
            max_count: Maximum number of commits to retrieve
            
        Returns:
            List of commits
        """
        if not self.is_git_repo():
            return [{"error": "Not a Git repository"}]
        
        try:
            commits = []
            for commit in self.repo.iter_commits(max_count=max_count):
                commits.append({
                    "hash": commit.hexsha,
                    "short_hash": commit.hexsha[:7],
                    "author": f"{commit.author.name} <{commit.author.email}>",
                    "date": commit.committed_datetime.isoformat(),
                    "message": commit.message
                })
            
            return commits
        except Exception as e:
            logger.error(f"Failed to get commit history: {e}")
            return [{"error": str(e)}]