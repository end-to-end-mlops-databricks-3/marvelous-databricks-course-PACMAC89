"""Dataloader fixture."""

import pandas as pd
import pytest
from loguru import logger
from pyspark.sql import SparkSession

from house_price import PROJECT_DIR
from house_price.config import ProjectConfig, Tags
from tests.unit_tests.spark_config import spark_config


@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    """Create and return a local SparkSession for testing."""
    spark = (
        SparkSession.builder.master(spark_config.master)  # Use the master configuration (e.g., "local[*]")
        .appName(spark_config.app_name)
        .config("spark.executor.cores", spark_config.spark_executor_cores)
        .config("spark.executor.instances", spark_config.spark_executor_instances)
        .config("spark.sql.shuffle.partitions", spark_config.spark_sql_shuffle_partitions)
        .config("spark.driver.bindAddress", spark_config.spark_driver_bindAddress)
        .getOrCreate()
    )

    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    """Load and return the project configuration."""
    config_file_path = (PROJECT_DIR / "project_config.yml").resolve()
    logger.info(f"Current config file path: {config_file_path.as_posix()}")
    config = ProjectConfig.from_yaml(config_file_path.as_posix())
    return config


@pytest.fixture(scope="function")
def sample_data(config: ProjectConfig, spark_session: SparkSession) -> pd.DataFrame:
    """Create a sample DataFrame from a CSV file."""
    file_path = PROJECT_DIR / "tests" / "test_data" / "sample.csv"
    sample = pd.read_csv(file_path.as_posix())
    return sample


@pytest.fixture(scope="session")
def tags() -> Tags:
    """Create and return a Tags instance for the test session."""
    return Tags(git_sha="wxyz", branch="test", job_run_id="9")
