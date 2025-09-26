// Titanic Survival Predictor App
class TitanicPredictor {
    constructor() {
        // Model coefficients from the trained logistic regression model
        this.coefficients = {
            'Pclass_3': -0.8356,
            'Sex_Male': -0.5930,
            'Pclass_1': 0.5454,
            'Has_Cabin': -0.3938,
            'Pclass_2': 0.3206,
            'Fare_Medium': 0.2935,
            'Age_Senior': 0.2572,
            'Fare_Very_High': 0.1808,
            'Embarked_S': -0.1348,
            'Age_Child': -0.1182,
            'Embarked_C': 0.0456,
            'Fare_High': 0.0987,
            'Age_Teen': -0.0234,
            'Age_Adult': 0.0123,
            'Age_Middle_Age': 0.0345,
            'Family_Size': 0.0156,
            'Is_Alone': -0.0678,
            'SibSp': -0.0234,
            'Parch': -0.0198,
            'Age': -0.0089,
            'Fare': 0.2381,
            'Embarked_Q': 0.0234
        };

        // Feature scaling parameters
        this.scaling = {
            'Age': { mean: 29.2, std: 13.4 },
            'Fare': { mean: 33.1, std: 46.8 },
            'SibSp': { mean: 0.43, std: 0.83 },
            'Parch': { mean: 0.40, std: 0.83 },
            'Family_Size': { mean: 1.83, std: 1.12 }
        };

        // Example passengers data
        this.examplePassengers = {
            rose: {
                name: "Rose DeWitt Bukater",
                age: 22,
                gender: "female",
                pclass: "1",
                fare: 150,
                sibsp: 0,
                parch: 1,
                embarked: "S",
                hasCabin: true
            },
            jack: {
                name: "Jack Dawson",
                age: 25,
                gender: "male",
                pclass: "3",
                fare: 7.5,
                sibsp: 0,
                parch: 0,
                embarked: "S",
                hasCabin: false
            },
            emma: {
                name: "Little Emma",
                age: 8,
                gender: "female",
                pclass: "2",
                fare: 25,
                sibsp: 0,
                parch: 2,
                embarked: "C",
                hasCabin: true
            }
        };

        this.init();
    }

    init() {
        this.bindEvents();
        this.setupFormValidation();
    }

    bindEvents() {
        // Form submission
        document.getElementById('predictionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handlePrediction();
        });

        // Reset form
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.resetForm();
        });

        // Try another prediction
        document.getElementById('tryAnotherBtn').addEventListener('click', () => {
            this.showPredictionForm();
        });

        // Example passengers
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const passengerKey = e.currentTarget.dataset.passenger;
                this.loadExamplePassenger(passengerKey);
            });
        });
    }

    setupFormValidation() {
        // Add real-time validation
        const form = document.getElementById('predictionForm');
        const inputs = form.querySelectorAll('input, select');
        
        inputs.forEach(input => {
            input.addEventListener('blur', () => {
                this.validateField(input);
            });
        });
    }

    validateField(field) {
        const value = field.value.trim();
        let isValid = true;

        // Remove previous error styling
        field.classList.remove('error');

        // Basic validation
        if (field.hasAttribute('required') && !value) {
            isValid = false;
        }

        // Specific validations
        if (field.type === 'number') {
            const min = parseFloat(field.min);
            const max = parseFloat(field.max);
            const numValue = parseFloat(value);

            if (value && (numValue < min || numValue > max)) {
                isValid = false;
            }
        }

        if (!isValid) {
            field.classList.add('error');
        }

        return isValid;
    }

    loadExamplePassenger(key) {
        const passenger = this.examplePassengers[key];
        if (!passenger) return;

        // Fill form with example data
        document.getElementById('passengerName').value = passenger.name;
        document.getElementById('age').value = passenger.age;
        document.querySelector(`input[name="gender"][value="${passenger.gender}"]`).checked = true;
        document.getElementById('pclass').value = passenger.pclass;
        document.getElementById('fare').value = passenger.fare;
        document.getElementById('sibsp').value = passenger.sibsp;
        document.getElementById('parch').value = passenger.parch;
        document.getElementById('embarked').value = passenger.embarked;
        document.getElementById('hasCabin').checked = passenger.hasCabin;

        // Add visual feedback
        this.showFormFilledFeedback();
    }

    showFormFilledFeedback() {
        const form = document.getElementById('predictionForm');
        form.style.transform = 'scale(1.02)';
        setTimeout(() => {
            form.style.transform = 'scale(1)';
        }, 200);
    }

    async handlePrediction() {
        const formData = this.getFormData();
        
        if (!this.validateForm(formData)) {
            this.showError('Please fill in all required fields correctly.');
            return;
        }

        // Show loading state
        this.showLoadingState();

        // Simulate processing time for better UX
        await this.delay(1500);

        try {
            const prediction = this.predict(formData);
            this.showResults(formData, prediction);
        } catch (error) {
            console.error('Prediction error:', error);
            this.showError('An error occurred during prediction. Please try again.');
        } finally {
            this.hideLoadingState();
        }
    }

    getFormData() {
        return {
            name: document.getElementById('passengerName').value.trim(),
            age: parseInt(document.getElementById('age').value),
            gender: document.querySelector('input[name="gender"]:checked')?.value,
            pclass: parseInt(document.getElementById('pclass').value),
            fare: parseFloat(document.getElementById('fare').value),
            sibsp: parseInt(document.getElementById('sibsp').value),
            parch: parseInt(document.getElementById('parch').value),
            embarked: document.getElementById('embarked').value,
            hasCabin: document.getElementById('hasCabin').checked
        };
    }

    validateForm(data) {
        return data.name && data.age && data.gender && data.pclass && 
               data.fare >= 0 && data.sibsp >= 0 && data.parch >= 0 && data.embarked;
    }

    predict(data) {
        // Calculate derived features
        const familySize = data.sibsp + data.parch + 1;
        const isAlone = familySize === 1;

        // Create age group
        let ageGroup;
        if (data.age <= 12) ageGroup = 'Child';
        else if (data.age <= 18) ageGroup = 'Teen';
        else if (data.age <= 30) ageGroup = 'Adult';
        else if (data.age <= 50) ageGroup = 'Middle_Age';
        else ageGroup = 'Senior';

        // Create fare group
        let fareGroup;
        if (data.fare <= 7.91) fareGroup = 'Low';
        else if (data.fare <= 14.45) fareGroup = 'Medium';
        else if (data.fare <= 31) fareGroup = 'High';
        else fareGroup = 'Very_High';

        // Scale numerical features
        const scaledAge = (data.age - this.scaling.Age.mean) / this.scaling.Age.std;
        const scaledFare = (data.fare - this.scaling.Fare.mean) / this.scaling.Fare.std;
        const scaledSibSp = (data.sibsp - this.scaling.SibSp.mean) / this.scaling.SibSp.std;
        const scaledParch = (data.parch - this.scaling.Parch.mean) / this.scaling.Parch.std;
        const scaledFamilySize = (familySize - this.scaling.Family_Size.mean) / this.scaling.Family_Size.std;

        // Calculate prediction using logistic regression
        let logit = 0;

        // Add coefficients for categorical features
        if (data.pclass === 1) logit += this.coefficients.Pclass_1;
        if (data.pclass === 2) logit += this.coefficients.Pclass_2;
        if (data.pclass === 3) logit += this.coefficients.Pclass_3;

        if (data.gender === 'male') logit += this.coefficients.Sex_Male;

        if (data.hasCabin) logit += this.coefficients.Has_Cabin;

        if (data.embarked === 'S') logit += this.coefficients.Embarked_S;
        if (data.embarked === 'C') logit += this.coefficients.Embarked_C;
        if (data.embarked === 'Q') logit += this.coefficients.Embarked_Q;

        // Age group
        if (ageGroup === 'Child') logit += this.coefficients.Age_Child;
        if (ageGroup === 'Teen') logit += this.coefficients.Age_Teen;
        if (ageGroup === 'Adult') logit += this.coefficients.Age_Adult;
        if (ageGroup === 'Middle_Age') logit += this.coefficients.Age_Middle_Age;
        if (ageGroup === 'Senior') logit += this.coefficients.Age_Senior;

        // Fare group
        if (fareGroup === 'Medium') logit += this.coefficients.Fare_Medium;
        if (fareGroup === 'High') logit += this.coefficients.Fare_High;
        if (fareGroup === 'Very_High') logit += this.coefficients.Fare_Very_High;

        // Numerical features
        logit += this.coefficients.Age * scaledAge;
        logit += this.coefficients.Fare * scaledFare;
        logit += this.coefficients.SibSp * scaledSibSp;
        logit += this.coefficients.Parch * scaledParch;
        logit += this.coefficients.Family_Size * scaledFamilySize;

        if (isAlone) logit += this.coefficients.Is_Alone;

        // Convert to probability using sigmoid function
        const probability = 1 / (1 + Math.exp(-logit));

        return {
            probability: probability,
            survived: probability > 0.5,
            factors: this.getKeyFactors(data, familySize, isAlone, ageGroup, fareGroup),
            details: {
                familySize,
                isAlone,
                ageGroup,
                fareGroup
            }
        };
    }

    getKeyFactors(data, familySize, isAlone, ageGroup, fareGroup) {
        const factors = [];

        // Gender impact
        if (data.gender === 'female') {
            factors.push({ name: 'Female Gender', impact: 'positive', description: 'Women had higher survival rates' });
        } else {
            factors.push({ name: 'Male Gender', impact: 'negative', description: 'Men had lower survival rates' });
        }

        // Class impact
        if (data.pclass === 1) {
            factors.push({ name: '1st Class', impact: 'positive', description: 'First-class passengers had better survival rates' });
        } else if (data.pclass === 3) {
            factors.push({ name: '3rd Class', impact: 'negative', description: 'Third-class passengers had lower survival rates' });
        }

        // Age impact
        if (ageGroup === 'Child') {
            factors.push({ name: 'Child Age', impact: 'positive', description: 'Children were prioritized in evacuations' });
        } else if (ageGroup === 'Senior') {
            factors.push({ name: 'Senior Age', impact: 'negative', description: 'Elderly passengers faced more challenges' });
        }

        // Cabin impact
        if (data.hasCabin) {
            factors.push({ name: 'Had Cabin', impact: 'positive', description: 'Cabin location improved escape chances' });
        } else {
            factors.push({ name: 'No Cabin', impact: 'negative', description: 'No assigned cabin reduced survival chances' });
        }

        // Fare impact
        if (fareGroup === 'Very_High') {
            factors.push({ name: 'High Fare', impact: 'positive', description: 'Higher fares indicated better accommodations' });
        } else if (fareGroup === 'Low') {
            factors.push({ name: 'Low Fare', impact: 'negative', description: 'Lower fares indicated poorer accommodations' });
        }

        // Family impact
        if (isAlone) {
            factors.push({ name: 'Traveling Alone', impact: 'negative', description: 'Solo travelers had slightly lower survival rates' });
        } else if (familySize > 4) {
            factors.push({ name: 'Large Family', impact: 'negative', description: 'Very large families faced coordination challenges' });
        }

        return factors.slice(0, 5); // Return top 5 factors
    }

    showResults(formData, prediction) {
        // Hide prediction form and show results
        document.querySelector('.prediction-section').style.display = 'none';
        document.getElementById('resultsSection').classList.remove('hidden');

        // Update survival indicator
        const indicator = document.getElementById('survivalIndicator');
        const icon = document.getElementById('survivalIcon');
        const text = document.getElementById('survivalText');

        if (prediction.survived) {
            indicator.className = 'survival-indicator survived';
            icon.textContent = '✅';
            text.textContent = 'LIKELY TO SURVIVE';
        } else {
            indicator.className = 'survival-indicator not-survived';
            icon.textContent = '❌';
            text.textContent = 'UNLIKELY TO SURVIVE';
        }

        // Update probability bar with animation
        const probabilityPercentage = Math.round(prediction.probability * 100);
        document.getElementById('probabilityPercentage').textContent = `${probabilityPercentage}%`;
        
        setTimeout(() => {
            document.getElementById('probabilityFill').style.width = `${probabilityPercentage}%`;
        }, 300);

        // Update passenger summary
        this.updatePassengerSummary(formData, prediction.details);

        // Update key factors
        this.updateKeyFactors(prediction.factors);

        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }

    updatePassengerSummary(data, details) {
        const summaryGrid = document.getElementById('summaryGrid');
        
        const summaryItems = [
            { label: 'Name', value: data.name },
            { label: 'Age', value: `${data.age} years (${details.ageGroup})` },
            { label: 'Gender', value: data.gender.charAt(0).toUpperCase() + data.gender.slice(1) },
            { label: 'Class', value: this.getClassLabel(data.pclass) },
            { label: 'Fare', value: `£${data.fare}` },
            { label: 'Family Size', value: details.familySize },
            { label: 'Embarked', value: this.getPortLabel(data.embarked) },
            { label: 'Had Cabin', value: data.hasCabin ? 'Yes' : 'No' }
        ];

        summaryGrid.innerHTML = summaryItems.map(item => `
            <div class="summary-item">
                <span class="summary-label">${item.label}:</span>
                <span class="summary-value">${item.value}</span>
            </div>
        `).join('');
    }

    updateKeyFactors(factors) {
        const factorsList = document.getElementById('factorsList');
        
        factorsList.innerHTML = factors.map(factor => `
            <div class="factor-item">
                <span class="factor-name">${factor.name}</span>
                <span class="factor-impact ${factor.impact}">
                    ${factor.impact === 'positive' ? '+' : factor.impact === 'negative' ? '-' : '~'}
                </span>
            </div>
        `).join('');
    }

    getClassLabel(pclass) {
        const labels = { 1: '1st Class', 2: '2nd Class', 3: '3rd Class' };
        return labels[pclass] || 'Unknown';
    }

    getPortLabel(embarked) {
        const labels = { 'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown' };
        return labels[embarked] || 'Unknown';
    }

    showPredictionForm() {
        document.querySelector('.prediction-section').style.display = 'block';
        document.getElementById('resultsSection').classList.add('hidden');
        document.querySelector('.prediction-section').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }

    resetForm() {
        document.getElementById('predictionForm').reset();
        document.getElementById('age').value = 30;
        document.getElementById('fare').value = 32;
        document.getElementById('sibsp').value = 0;
        document.getElementById('parch').value = 0;
        
        // Remove any error styling
        document.querySelectorAll('.form-control').forEach(field => {
            field.classList.remove('error');
        });
    }

    showLoadingState() {
        const btn = document.getElementById('predictBtn');
        btn.disabled = true;
        btn.querySelector('.btn-text').style.display = 'none';
        btn.querySelector('.loading-spinner').classList.remove('hidden');
    }

    hideLoadingState() {
        const btn = document.getElementById('predictBtn');
        btn.disabled = false;
        btn.querySelector('.btn-text').style.display = 'inline-block';
        btn.querySelector('.loading-spinner').classList.add('hidden');
    }

    showError(message) {
        // Simple error display - could be enhanced with a proper modal
        alert(message);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TitanicPredictor();
    
    // Add some nice touches
    addInteractiveEffects();
});

function addInteractiveEffects() {
    // Add hover effects to cards
    document.querySelectorAll('.card').forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-2px)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
        });
    });

    // Add focus effects to form controls
    document.querySelectorAll('.form-control').forEach(control => {
        control.addEventListener('focus', () => {
            control.parentElement.style.transform = 'scale(1.01)';
        });
        
        control.addEventListener('blur', () => {
            control.parentElement.style.transform = 'scale(1)';
        });
    });
}