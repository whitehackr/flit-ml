"""
Redis-based prediction storage implementation

Provides Redis caching for ML predictions with batch upload queuing for BigQuery.
Integrates with Data Engineering team's Redis infrastructure for production deployment.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import redis
from dataclasses import asdict

from flit_ml.core.shadow_controller import PredictionLog


class RedisPredictionStorage:
    """
    Redis-based storage implementation for prediction logging.

    Provides real-time caching with batch upload queuing for Data Engineering's
    BigQuery upload pipeline.
    """

    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 prediction_ttl: int = 2592000,  # 30 days
                 upload_queue_name: str = "ml_prediction_upload_queue",
                 key_prefix: str = "ml:bnpl",
                 verbose: bool = False):
        """
        Initialize Redis storage.

        Args:
            redis_client: Redis client instance (if None, creates default)
            prediction_ttl: TTL for prediction records in seconds (default: 30 days)
            upload_queue_name: Redis list name for batch upload queue
            key_prefix: Prefix for all Redis keys
            verbose: Enable verbose logging
        """
        self.redis_client = redis_client or self._create_default_client()
        self.prediction_ttl = prediction_ttl
        self.upload_queue_name = upload_queue_name
        self.key_prefix = key_prefix
        self.verbose = verbose

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # Test Redis connection
        self._test_connection()

    def _create_default_client(self) -> redis.Redis:
        """Create default Redis client with fallback configuration."""
        # Default configuration - should be overridden with environment variables
        return redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True
        )

    def _test_connection(self):
        """Test Redis connection and log status."""
        try:
            self.redis_client.ping()
            if self.verbose:
                print(f"✅ Redis connection successful")
        except redis.ConnectionError as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise

    def _prediction_key(self, prediction_id: str) -> str:
        """Generate Redis key for prediction."""
        return f"{self.key_prefix}:pred:{prediction_id}"

    def _transaction_key(self, transaction_id: str) -> str:
        """Generate Redis key for transaction data."""
        return f"{self.key_prefix}:tx:{transaction_id}"

    def _experiment_key(self, experiment_id: str) -> str:
        """Generate Redis key for experiment data."""
        return f"{self.key_prefix}:exp:{experiment_id}"

    def store_prediction(self, prediction_log: PredictionLog) -> None:
        """
        Store prediction log in Redis with TTL and queue for batch upload.

        Args:
            prediction_log: Prediction log to store
        """
        try:
            # Convert prediction log to JSON
            prediction_data = asdict(prediction_log)
            # Handle datetime serialization
            prediction_data['timestamp'] = prediction_log.timestamp.isoformat()
            prediction_json = json.dumps(prediction_data)

            # Store prediction with TTL
            prediction_key = self._prediction_key(prediction_log.prediction_id)
            self.redis_client.setex(
                prediction_key,
                self.prediction_ttl,
                prediction_json
            )

            # Add to upload queue for Data Engineering batch processing
            upload_record = {
                "type": "prediction",
                "key": prediction_key,
                "prediction_id": prediction_log.prediction_id,
                "timestamp": prediction_log.timestamp.isoformat(),
                "experiment_id": prediction_log.experiment_id
            }
            self.redis_client.lpush(self.upload_queue_name, json.dumps(upload_record))

            if self.verbose:
                print(f"📦 Stored prediction {prediction_log.prediction_id} in Redis")

        except redis.ConnectionError as e:
            self.logger.error(f"Redis storage failed for prediction {prediction_log.prediction_id}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error storing prediction {prediction_log.prediction_id}: {e}")
            raise

    def get_recent_predictions(self, hours: int) -> List[PredictionLog]:
        """
        Retrieve recent predictions from Redis.

        Args:
            hours: Number of hours back to retrieve

        Returns:
            List of recent prediction logs
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            predictions = []

            # Use Redis SCAN to iterate through prediction keys
            cursor = 0
            pattern = f"{self.key_prefix}:pred:*"

            while True:
                cursor, keys = self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )

                for key in keys:
                    try:
                        prediction_json = self.redis_client.get(key)
                        if prediction_json:
                            prediction_data = json.loads(prediction_json)

                            # Parse timestamp
                            timestamp = datetime.fromisoformat(prediction_data['timestamp'])

                            # Filter by time
                            if timestamp >= cutoff_time:
                                # Convert back to PredictionLog
                                prediction_data['timestamp'] = timestamp
                                prediction_log = PredictionLog(**prediction_data)
                                predictions.append(prediction_log)

                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        self.logger.warning(f"Failed to parse prediction from key {key}: {e}")
                        continue

                if cursor == 0:
                    break

            # Sort by timestamp (most recent first)
            predictions.sort(key=lambda x: x.timestamp, reverse=True)

            if self.verbose:
                print(f"📊 Retrieved {len(predictions)} predictions from last {hours} hours")

            return predictions

        except redis.ConnectionError as e:
            self.logger.error(f"Redis connection failed during retrieval: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving recent predictions: {e}")
            return []

    def get_experiment_data(self, experiment_id: str) -> List[PredictionLog]:
        """
        Get all predictions for a specific experiment.

        Args:
            experiment_id: Experiment ID to filter by

        Returns:
            List of predictions for the experiment
        """
        try:
            predictions = []

            # Use Redis SCAN to find predictions with matching experiment_id
            cursor = 0
            pattern = f"{self.key_prefix}:pred:*"

            while True:
                cursor, keys = self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )

                for key in keys:
                    try:
                        prediction_json = self.redis_client.get(key)
                        if prediction_json:
                            prediction_data = json.loads(prediction_json)

                            # Filter by experiment ID
                            if prediction_data.get('experiment_id') == experiment_id:
                                # Parse timestamp
                                prediction_data['timestamp'] = datetime.fromisoformat(prediction_data['timestamp'])
                                prediction_log = PredictionLog(**prediction_data)
                                predictions.append(prediction_log)

                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        self.logger.warning(f"Failed to parse prediction from key {key}: {e}")
                        continue

                if cursor == 0:
                    break

            # Sort by timestamp
            predictions.sort(key=lambda x: x.timestamp)

            if self.verbose:
                print(f"🧪 Retrieved {len(predictions)} predictions for experiment {experiment_id}")

            return predictions

        except redis.ConnectionError as e:
            self.logger.error(f"Redis connection failed during experiment retrieval: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving experiment data: {e}")
            return []

    def store_transaction_data(self, transaction_id: str, transaction_data: Dict[str, Any]) -> None:
        """
        Store transaction data for later correlation with predictions.

        Args:
            transaction_id: Transaction identifier
            transaction_data: Complete transaction data
        """
        try:
            transaction_key = self._transaction_key(transaction_id)
            transaction_json = json.dumps(transaction_data)

            self.redis_client.setex(
                transaction_key,
                self.prediction_ttl,
                transaction_json
            )

            # Add to upload queue
            upload_record = {
                "type": "transaction",
                "key": transaction_key,
                "transaction_id": transaction_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.redis_client.lpush(self.upload_queue_name, json.dumps(upload_record))

            if self.verbose:
                print(f"💳 Stored transaction {transaction_id} in Redis")

        except Exception as e:
            self.logger.error(f"Failed to store transaction {transaction_id}: {e}")
            raise

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get status information about the upload queue.

        Returns:
            Dictionary with queue status information
        """
        try:
            queue_length = self.redis_client.llen(self.upload_queue_name)

            # Get Redis info
            redis_info = self.redis_client.info()

            return {
                "upload_queue_length": queue_length,
                "redis_connected_clients": redis_info.get("connected_clients", 0),
                "redis_used_memory_human": redis_info.get("used_memory_human", "unknown"),
                "redis_uptime_in_seconds": redis_info.get("uptime_in_seconds", 0)
            }

        except Exception as e:
            self.logger.error(f"Failed to get queue status: {e}")
            return {"error": "Failed to retrieve queue status"}

    def cleanup_expired_data(self, dry_run: bool = True) -> Dict[str, int]:
        """
        Cleanup expired prediction data (manual cleanup).

        Args:
            dry_run: If True, only count what would be deleted

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            expired_count = 0
            total_scanned = 0

            cursor = 0
            pattern = f"{self.key_prefix}:pred:*"

            while True:
                cursor, keys = self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )

                for key in keys:
                    total_scanned += 1

                    # Check if key has TTL
                    ttl = self.redis_client.ttl(key)

                    if ttl == -1:  # No TTL set
                        expired_count += 1
                        if not dry_run:
                            self.redis_client.delete(key)

                if cursor == 0:
                    break

            action = "would delete" if dry_run else "deleted"

            if self.verbose:
                print(f"🧹 Cleanup: {action} {expired_count}/{total_scanned} keys")

            return {
                "total_scanned": total_scanned,
                "expired_count": expired_count,
                "action": action
            }

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return {"error": "Cleanup failed"}

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Redis storage.

        Returns:
            Dictionary with health status
        """
        try:
            start_time = time.time()

            # Test basic operations
            test_key = f"{self.key_prefix}:health_check"
            test_value = json.dumps({"test": True, "timestamp": datetime.utcnow().isoformat()})

            # Test write
            self.redis_client.setex(test_key, 60, test_value)

            # Test read
            retrieved_value = self.redis_client.get(test_key)

            # Test delete
            self.redis_client.delete(test_key)

            response_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "operations_tested": ["write", "read", "delete"],
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }