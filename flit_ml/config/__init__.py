"""
Configuration module for Flit ML.

Provides secure, multi-environment configuration management
for BigQuery and other external services.
"""

from .bigquery import BigQueryConfig, config

__all__ = ["BigQueryConfig", "config"]