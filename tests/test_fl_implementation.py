"""
Unit tests for Federated Learning Personal Finance Implementation
"""
import pytest
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fl_personal_finance_implementation import (
    PersonalFinanceDataset,
    SavingsClassifier,
    ClientConfig,
    AdaptiveFedProxClient,
    CityTierPartitioner,
    OccupationPartitioner,
    IncomeBracketPartitioner,
    DirichletPartitioner,
    create_client_dataloaders
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(1000, 15)
    y = np.random.randint(0, 3, 1000)
    return X, y


@pytest.fixture
def sample_metadata():
    """Create sample metadata"""
    return {
        'num_features': 15,
        'num_classes': 3,
        'class_names': ['Low Savings', 'Medium Savings', 'High Savings'],
        'original_df': pd.DataFrame({
            'City_Tier': np.random.choice(['Tier_1', 'Tier_2', 'Tier_3'], 1000),
            'Occupation': np.random.choice(['Professional', 'Retired', 'Self_Employed', 'Student'], 1000),
            'Income': np.random.uniform(10000, 100000, 1000)
        })
    }


@pytest.fixture
def sample_model():
    """Create sample model"""
    return SavingsClassifier(input_dim=15, num_classes=3, hidden_dims=[64, 32])


@pytest.fixture
def sample_config():
    """Create sample client configuration"""
    return ClientConfig(
        mu=1.0,
        lr=0.01,
        local_epochs=2,
        adaptive_mu=False,
        use_fedbn=False
    )


# ============================================================================
# DATASET TESTS
# ============================================================================

class TestPersonalFinanceDataset:
    """Test cases for PersonalFinanceDataset"""
    
    def test_dataset_creation(self, sample_data):
        """Test dataset initialization"""
        X, y = sample_data
        dataset = PersonalFinanceDataset(X, y)
        
        assert len(dataset) == len(y)
        assert isinstance(dataset.features, torch.Tensor)
        assert isinstance(dataset.labels, torch.Tensor)
    
    def test_dataset_getitem(self, sample_data):
        """Test dataset item retrieval"""
        X, y = sample_data
        dataset = PersonalFinanceDataset(X, y)
        
        features, label = dataset[0]
        assert features.shape == (15,)
        assert isinstance(label, torch.Tensor)
    
    def test_dataset_length(self, sample_data):
        """Test dataset length"""
        X, y = sample_data
        dataset = PersonalFinanceDataset(X, y)
        assert len(dataset) == 1000


# ============================================================================
# MODEL TESTS
# ============================================================================

class TestSavingsClassifier:
    """Test cases for SavingsClassifier model"""
    
    def test_model_creation(self):
        """Test model initialization"""
        model = SavingsClassifier(input_dim=15, num_classes=3, hidden_dims=[64, 32])
        assert model is not None
        assert hasattr(model, 'feature_extractor')
        assert hasattr(model, 'classifier')
    
    def test_model_forward_pass(self, sample_model):
        """Test forward pass"""
        x = torch.randn(10, 15)
        output = sample_model(x)
        
        assert output.shape == (10, 3)
    
    def test_model_with_batch_norm(self):
        """Test model with batch normalization"""
        model = SavingsClassifier(input_dim=15, num_classes=3, use_bn=True)
        x = torch.randn(10, 15)
        output = model(x)
        
        assert output.shape == (10, 3)
    
    def test_model_without_batch_norm(self):
        """Test model without batch normalization"""
        model = SavingsClassifier(input_dim=15, num_classes=3, use_bn=False)
        x = torch.randn(10, 15)
        output = model(x)
        
        assert output.shape == (10, 3)
    
    def test_model_parameter_count(self, sample_model):
        """Test model has trainable parameters"""
        params = list(sample_model.parameters())
        assert len(params) > 0


# ============================================================================
# PARTITIONING TESTS
# ============================================================================

class TestPartitioners:
    """Test cases for data partitioning strategies"""
    
    def test_city_tier_partitioner(self, sample_data, sample_metadata):
        """Test city tier based partitioning"""
        X, y = sample_data
        partitioner = CityTierPartitioner()
        partitions = partitioner.partition(X, y, sample_metadata, 14)
        
        assert len(partitions) == 14
        assert all(len(p) > 0 for p in partitions)
        
        # Check all indices are used
        all_indices = [idx for partition in partitions for idx in partition]
        assert len(set(all_indices)) == len(all_indices)  # No duplicates
    
    def test_dirichlet_partitioner(self, sample_data, sample_metadata):
        """Test Dirichlet distribution partitioning"""
        X, y = sample_data
        partitioner = DirichletPartitioner(alpha=0.5)
        partitions = partitioner.partition(X, y, sample_metadata, 10)
        
        assert len(partitions) == 10
        assert all(len(p) > 0 for p in partitions)
        
        # Check total samples preserved
        total_samples = sum(len(p) for p in partitions)
        assert total_samples == len(y)
    
    def test_occupation_partitioner(self, sample_data, sample_metadata):
        """Test occupation based partitioning"""
        X, y = sample_data
        partitioner = OccupationPartitioner()
        partitions = partitioner.partition(X, y, sample_metadata, 16)
        
        assert len(partitions) > 0
    
    def test_income_bracket_partitioner(self, sample_data, sample_metadata):
        """Test income bracket partitioning"""
        X, y = sample_data
        partitioner = IncomeBracketPartitioner()
        partitions = partitioner.partition(X, y, sample_metadata, 10)
        
        assert len(partitions) == 10
        assert all(len(p) > 0 for p in partitions)


# ============================================================================
# CLIENT TESTS
# ============================================================================

class TestAdaptiveFedProxClient:
    """Test cases for AdaptiveFedProxClient"""
    
    def test_client_creation(self, sample_model, sample_data, sample_config):
        """Test client initialization"""
        X, y = sample_data
        dataset = PersonalFinanceDataset(X[:100], y[:100])
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        client = AdaptiveFedProxClient(
            model=sample_model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=sample_config
        )
        
        assert client is not None
        assert client.mu == 1.0
    
    def test_client_get_parameters(self, sample_model, sample_data, sample_config):
        """Test getting client parameters"""
        X, y = sample_data
        dataset = PersonalFinanceDataset(X[:100], y[:100])
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        client = AdaptiveFedProxClient(
            model=sample_model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=sample_config
        )
        
        params = client.get_parameters()
        assert len(params) > 0
        assert all(isinstance(p, np.ndarray) for p in params)
    
    def test_client_set_parameters(self, sample_model, sample_data, sample_config):
        """Test setting client parameters"""
        X, y = sample_data
        dataset = PersonalFinanceDataset(X[:100], y[:100])
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        client = AdaptiveFedProxClient(
            model=sample_model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=sample_config
        )
        
        initial_params = client.get_parameters()
        new_params = [p + 0.1 for p in initial_params]
        client.set_parameters(new_params)
        updated_params = client.get_parameters()
        
        assert not np.allclose(initial_params[0], updated_params[0])
    
    @pytest.mark.slow
    def test_client_train(self, sample_model, sample_data, sample_config):
        """Test client training"""
        X, y = sample_data
        dataset = PersonalFinanceDataset(X[:100], y[:100])
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        client = AdaptiveFedProxClient(
            model=sample_model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=sample_config
        )
        
        global_params = client.get_parameters()
        metrics = client.train(global_params)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'mu' in metrics
        assert metrics['loss'] >= 0
        assert 0 <= metrics['accuracy'] <= 100
    
    def test_client_evaluate(self, sample_model, sample_data, sample_config):
        """Test client evaluation"""
        X, y = sample_data
        dataset = PersonalFinanceDataset(X[:100], y[:100])
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        client = AdaptiveFedProxClient(
            model=sample_model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=sample_config
        )
        
        metrics = client.evaluate()
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert metrics['loss'] >= 0
        assert 0 <= metrics['accuracy'] <= 100


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestEndToEnd:
    """Integration tests for complete workflow"""
    
    @pytest.mark.slow
    def test_complete_workflow(self, sample_data, sample_metadata):
        """Test complete FL workflow"""
        X, y = sample_data
        
        # Partition data
        partitioner = DirichletPartitioner(alpha=0.5)
        partitions = partitioner.partition(X, y, sample_metadata, 5)
        
        # Create dataloaders
        client_loaders = create_client_dataloaders(X, y, partitions, batch_size=16)
        
        # Create clients
        config = ClientConfig(mu=0.1, lr=0.01, local_epochs=1)
        clients = []
        
        for loader_dict in client_loaders:
            model = SavingsClassifier(
                input_dim=15,
                num_classes=3,
                hidden_dims=[32, 16]
            )
            client = AdaptiveFedProxClient(
                model=model,
                train_loader=loader_dict['train'],
                test_loader=loader_dict['test'],
                config=config
            )
            clients.append(client)
        
        # Simulate one round
        global_params = clients[0].get_parameters()
        
        for client in clients:
            client.set_parameters(global_params)
            metrics = client.train(global_params)
            assert metrics['loss'] >= 0
        
        assert len(clients) == 5


# ============================================================================
# SMOKE TESTS
# ============================================================================

@pytest.mark.smoke
class TestSmoke:
    """Quick smoke tests for CI"""
    
    def test_imports(self):
        """Test that all imports work"""
        from fl_personal_finance_implementation import (
            PersonalFinanceDataset,
            SavingsClassifier,
            AdaptiveFedProxClient
        )
        assert True
    
    def test_basic_model_creation(self):
        """Test basic model can be created"""
        model = SavingsClassifier(input_dim=15, num_classes=3)
        assert model is not None
