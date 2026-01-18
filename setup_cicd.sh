#!/bin/bash
# CI/CD Setup Script for FL Personal Finance Project
# This script sets up the CI/CD pipeline and development environment

set -e  # Exit on error

echo "🚀 FL Personal Finance CI/CD Setup"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "Not a git repository. Please run this script from your repository root."
    exit 1
fi

print_success "Git repository detected"

# Create necessary directories
echo ""
echo "📁 Creating directory structure..."
mkdir -p .github/workflows
mkdir -p tests
mkdir -p results
mkdir -p artifacts
mkdir -p logs
mkdir -p experiments
mkdir -p scripts

print_success "Directories created"

# Check if GitHub Actions workflow exists
echo ""
echo "🔍 Checking for GitHub Actions workflow..."
if [ -f ".github/workflows/fl_ci_cd.yml" ]; then
    print_warning "GitHub Actions workflow already exists"
else
    print_error "GitHub Actions workflow not found"
    echo "Please copy the fl_ci_cd.yml file to .github/workflows/"
fi

# Check for Python
echo ""
echo "🐍 Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION detected"
else
    print_error "Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

# Check for required files
echo ""
echo "📄 Checking for required files..."
REQUIRED_FILES=(
    "requirements.txt"
    "pytest.ini"
    "pyproject.toml"
    ".gitignore"
    ".flake8"
    "Makefile"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "$file exists"
    else
        print_warning "$file not found - please create it"
    fi
done

# Offer to create virtual environment
echo ""
read -p "Would you like to create a virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created at ./venv"
    echo ""
    echo "To activate it, run:"
    echo "  source venv/bin/activate  (Linux/Mac)"
    echo "  venv\\Scripts\\activate     (Windows)"
fi

# Offer to install dependencies
echo ""
read -p "Would you like to install project dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "venv" ]; then
        source venv/bin/activate 2>/dev/null || true
    fi
    
    echo "Installing dependencies..."
    pip install --upgrade pip
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_error "requirements.txt not found"
    fi
fi

# Set up pre-commit hooks
echo ""
read -p "Would you like to set up pre-commit hooks? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "Makefile" ]; then
        make setup-hooks
        print_success "Pre-commit hooks installed"
    else
        echo "Creating pre-commit hook manually..."
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for FL Personal Finance

echo "Running pre-commit checks..."

# Format check
echo "Checking code formatting..."
black --check . || {
    echo "Code formatting check failed. Run 'make format' to fix."
    exit 1
}

# Linting
echo "Running linting..."
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || {
    echo "Linting failed. Fix the errors and try again."
    exit 1
}

# Quick tests
echo "Running quick tests..."
pytest tests/ -v -m "not slow" --tb=short || {
    echo "Tests failed. Fix the tests and try again."
    exit 1
}

echo "All pre-commit checks passed!"
EOF
        chmod +x .git/hooks/pre-commit
        print_success "Pre-commit hook created"
    fi
fi

# Check GitHub CLI
echo ""
echo "🔧 Checking for GitHub CLI..."
if command -v gh &> /dev/null; then
    print_success "GitHub CLI detected"
    
    read -p "Would you like to enable GitHub Actions for this repo? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gh workflow list
    fi
else
    print_warning "GitHub CLI not found. Install it to manage workflows from CLI:"
    echo "  https://cli.github.com/"
fi

# Create sample experiment script
echo ""
read -p "Would you like to create a sample experiment script? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat > experiments/run_baseline.py << 'EOF'
"""
Sample FL Experiment Script
Run with: python experiments/run_baseline.py
"""
import sys
sys.path.append('.')

from fl_personal_finance_implementation import *

def main():
    print("Running baseline FL experiment...")
    # Add your experiment code here
    
if __name__ == "__main__":
    main()
EOF
    print_success "Sample experiment script created at experiments/run_baseline.py"
fi

# Summary
echo ""
echo "═══════════════════════════════════"
echo "✅ Setup Complete!"
echo "═══════════════════════════════════"
echo ""
echo "Next steps:"
echo "1. Copy CI/CD files to your repository:"
echo "   - .github/workflows/fl_ci_cd.yml"
echo "   - requirements.txt"
echo "   - pytest.ini"
echo "   - pyproject.toml"
echo "   - .flake8"
echo "   - Makefile"
echo ""
echo "2. Add and commit these files:"
echo "   git add ."
echo "   git commit -m 'Add CI/CD pipeline'"
echo ""
echo "3. Push to GitHub:"
echo "   git push origin main"
echo ""
echo "4. Check GitHub Actions tab to see your workflow run!"
echo ""
echo "📚 For detailed documentation, see CI_CD_README.md"
echo ""
echo "🎯 Quick commands:"
echo "   make test        - Run all tests"
echo "   make lint        - Check code quality"
echo "   make format      - Auto-format code"
echo "   make ci-local    - Run CI checks locally"
echo ""
