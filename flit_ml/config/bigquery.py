"""
BigQuery configuration and connection management.

Handles multi-environment authentication with secure credential management:
1. GCP Application Default Credentials (production GCP)
2. Railway encrypted environment variables (Railway deployment)
3. Local service account file (development)
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import Tuple, Optional
from google.cloud import bigquery
from google.oauth2 import service_account
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError

logger = logging.getLogger(__name__)


class BigQueryAuthenticationError(Exception):
    """Raised when BigQuery authentication fails across all methods."""
    pass


class BigQueryConfig:
    """Configuration for BigQuery connections with multi-environment auth."""

    def __init__(self, project_id: str = "flit-data-platform"):
        self.project_id = project_id
        self._client: Optional[bigquery.Client] = None

    def _try_application_default_credentials(self) -> Tuple[Optional[object], Optional[str]]:
        """
        Try GCP Application Default Credentials.

        Works when:
        - Running on GCP (Cloud Run, Compute Engine)
        - gcloud CLI authenticated locally
        - GOOGLE_APPLICATION_CREDENTIALS env var set

        Returns:
            Tuple of (credentials, project_id) or (None, None)
        """
        try:
            credentials, project = default()
            logger.info("Using Application Default Credentials")
            return credentials, project or self.project_id
        except DefaultCredentialsError:
            logger.debug("Application Default Credentials not available")
            return None, None

    def _try_environment_variable(self) -> Tuple[Optional[object], Optional[str]]:
        """
        Try Railway encrypted environment variable.

        Expects GOOGLE_SERVICE_ACCOUNT_KEY as base64-encoded JSON.
        Use Railway's secret encryption:

        railway variables set --secret GOOGLE_SERVICE_ACCOUNT_KEY=<base64_key>

        Returns:
            Tuple of (credentials, project_id) or (None, None)
        """
        env_key = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY")
        if not env_key:
            logger.debug("GOOGLE_SERVICE_ACCOUNT_KEY environment variable not set")
            return None, None

        try:
            # Decode base64 environment variable
            key_data = json.loads(base64.b64decode(env_key))
            credentials = service_account.Credentials.from_service_account_info(key_data)
            project_id = key_data.get("project_id", self.project_id)

            logger.info("Using environment variable credentials (Railway)")
            return credentials, project_id
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse GOOGLE_SERVICE_ACCOUNT_KEY: {e}")
            return None, None

    def _try_local_service_account(self) -> Tuple[Optional[object], Optional[str]]:
        """
        Try local service account file (development only).

        Looks for flit-data-platform-dev-sa.json in standard location.

        Returns:
            Tuple of (credentials, project_id) or (None, None)
        """
        # Standard development path
        dev_path = Path.home() / "Documents" / "repos" / ".gcp" / "flit-data-platform-dev-sa.json"

        if not dev_path.exists():
            logger.debug(f"Local service account file not found: {dev_path}")
            return None, None

        try:
            credentials = service_account.Credentials.from_service_account_file(str(dev_path))
            logger.info(f"Using local service account file: {dev_path}")
            return credentials, self.project_id
        except Exception as e:
            logger.warning(f"Failed to load local service account: {e}")
            return None, None

    def get_credentials(self) -> Tuple[object, str]:
        """
        Get BigQuery credentials using multi-environment strategy.

        Authentication hierarchy:
        1. GCP Application Default Credentials (most secure)
        2. Railway environment variable (Railway deployment)
        3. Local service account file (development)

        Returns:
            Tuple of (credentials, project_id)

        Raises:
            BigQueryAuthenticationError: If no authentication method succeeds
        """
        # Try each authentication method in order
        auth_methods = [
            ("Application Default Credentials", self._try_application_default_credentials),
            ("Environment Variable", self._try_environment_variable),
            ("Local Service Account", self._try_local_service_account),
        ]

        for method_name, method_func in auth_methods:
            credentials, project_id = method_func()
            if credentials:
                logger.info(f"Successfully authenticated using: {method_name}")
                return credentials, project_id

        # All methods failed
        raise BigQueryAuthenticationError(
            "Failed to authenticate with BigQuery. Tried:\n"
            "1. GCP Application Default Credentials\n"
            "2. GOOGLE_SERVICE_ACCOUNT_KEY environment variable\n"
            "3. Local service account file\n\n"
            "For Railway deployment, set encrypted environment variable:\n"
            "railway variables set --secret GOOGLE_SERVICE_ACCOUNT_KEY=<base64_key>\n\n"
            "For local development, ensure service account file exists at:\n"
            f"{Path.home() / 'Documents' / 'repos' / '.gcp' / 'flit-data-platform-dev-sa.json'}"
        )

    def get_client(self) -> bigquery.Client:
        """
        Get authenticated BigQuery client.

        Returns:
            Configured BigQuery client

        Raises:
            BigQueryAuthenticationError: If authentication fails
        """
        if self._client is None:
            credentials, project_id = self.get_credentials()
            self._client = bigquery.Client(
                project=project_id,
                credentials=credentials
            )
            logger.info(f"BigQuery client created for project: {project_id}")

        return self._client

    def test_connection(self) -> bool:
        """
        Test BigQuery connection and permissions.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            client = self.get_client()
            # Simple query to test connection
            query = "SELECT 1 as test_value"
            result = client.query(query).result()

            # Verify we can read results
            for row in result:
                assert row.test_value == 1

            logger.info("BigQuery connection test successful")
            return True
        except Exception as e:
            logger.error(f"BigQuery connection test failed: {e}")
            return False


# Global configuration instance
config = BigQueryConfig()