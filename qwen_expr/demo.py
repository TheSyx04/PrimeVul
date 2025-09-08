#!/usr/bin/env python3
"""
Demo script for Qwen vulnerability detection.
Shows the complete workflow without requiring actual model loading.
"""

import json
import os


def create_demo_data():
    """Create demo test data."""
    demo_samples = [
        {
            "project": "demo_project_1",
            "commit_id": "commit123",
            "func": """void unsafe_copy(char *dest, char *src) {
    while (*src) {
        *dest++ = *src++;
    }
    *dest = '\\0';
}""",
            "target": "YES"
        },
        {
            "project": "demo_project_2", 
            "commit_id": "commit456",
            "func": """int safe_add(int a, int b) {
    if (a > 0 && b > INT_MAX - a) {
        return INT_MAX;  // Overflow protection
    }
    if (a < 0 && b < INT_MIN - a) {
        return INT_MIN;  // Underflow protection  
    }
    return a + b;
}""",
            "target": "NO"
        },
        {
            "project": "demo_project_3",
            "commit_id": "commit789", 
            "func": """char* get_user_input() {
    char buffer[100];
    gets(buffer);  // Dangerous function
    return strdup(buffer);
}""",
            "target": "YES"
        }
    ]
    
    demo_file = "demo_test_data.jsonl"
    with open(demo_file, 'w') as f:
        for sample in demo_samples:
            f.write(json.dumps(sample) + '\n')
    
    return demo_file, demo_samples


def simulate_qwen_responses():
    """Simulate what Qwen model responses might look like."""
    
    responses = [
        # Response for unsafe_copy function
        """1. **Code Understanding**: This function copies a null-terminated string from src to dest.

2. **Input Analysis**: Takes two char pointers - dest (destination buffer) and src (source string).

3. **Data Flow**: Copies characters one by one from src to dest until null terminator is found.

4. **Security Checks**: 
   - No bounds checking on destination buffer
   - No validation of buffer sizes
   - Classic buffer overflow vulnerability

5. **Bounds Checking**: CRITICAL ISSUE - No verification that dest buffer can accommodate the entire src string.

6. **Memory Management**: No memory allocation, but unsafe memory access possible.

7. **Input Validation**: No validation of input pointers or buffer sizes.

**Final Answer**: YES""",

        # Response for safe_add function  
        """1. **Code Understanding**: This function performs safe integer addition with overflow/underflow protection.

2. **Input Analysis**: Takes two integer parameters a and b.

3. **Data Flow**: Checks for potential overflow/underflow conditions before performing addition.

4. **Security Checks**: 
   - Proper overflow detection using INT_MAX
   - Proper underflow detection using INT_MIN
   - Safe bounds checking implemented

5. **Bounds Checking**: Excellent bounds checking for integer arithmetic.

6. **Memory Management**: No memory operations involved.

7. **Input Validation**: Comprehensive validation for arithmetic safety.

**Final Answer**: NO""",

        # Response for get_user_input function
        """1. **Code Understanding**: This function reads user input into a buffer and returns a copy.

2. **Input Analysis**: No direct parameters, reads from standard input.

3. **Data Flow**: Uses gets() to read into fixed buffer, then duplicates with strdup().

4. **Security Checks**: 
   - CRITICAL: gets() is inherently unsafe and deprecated
   - Buffer overflow vulnerability present
   - No input length validation

5. **Bounds Checking**: No bounds checking - gets() reads unlimited input into fixed buffer.

6. **Memory Management**: Uses strdup() which is safe, but buffer overflow occurs before this.

7. **Input Validation**: No input validation whatsoever.

**Final Answer**: YES"""
    ]
    
    return responses


def extract_demo_prediction(response):
    """Extract prediction from demo response."""
    if "**Final Answer**: YES" in response:
        return "YES"
    elif "**Final Answer**: NO" in response:
        return "NO"
    else:
        return "UNCLEAR"


def run_demo():
    """Run the complete demo workflow."""
    
    print("Qwen3-Coder-480B-A35B-Instruct Vulnerability Detection Demo")
    print("=" * 60)
    print()
    
    # Create demo data
    print("1. Creating demo test data...")
    demo_file, samples = create_demo_data()
    print(f"   Created {len(samples)} test samples in {demo_file}")
    print()
    
    # Simulate model responses
    print("2. Simulating Qwen model responses...")
    responses = simulate_qwen_responses()
    print(f"   Generated {len(responses)} model responses")
    print()
    
    # Process results
    print("3. Processing results...")
    results = []
    correct = 0
    
    for i, (sample, response) in enumerate(zip(samples, responses)):
        prediction = extract_demo_prediction(response)
        is_correct = prediction == sample["target"]
        if is_correct:
            correct += 1
            
        result = {
            "sample_key": f"{sample['project']}_{sample['commit_id']}",
            "target": sample["target"],
            "prediction": prediction,
            "correct": is_correct,
            "response": response
        }
        results.append(result)
        
        print(f"   Sample {i+1}: {sample['project']}")
        print(f"   Target: {sample['target']}, Predicted: {prediction}, Correct: {is_correct}")
        print()
    
    # Calculate accuracy
    accuracy = correct / len(samples)
    print(f"4. Results Summary:")
    print(f"   Total samples: {len(samples)}")
    print(f"   Correct predictions: {correct}")
    print(f"   Accuracy: {accuracy:.2%}")
    print()
    
    # Save results
    output_file = "demo_results.jsonl"
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, indent=2))
            f.write('\n')
    
    # Create predictions file for VD-Score
    pred_file = "demo_predictions.txt"
    with open(pred_file, 'w') as f:
        for result in results:
            f.write("1\n" if result["prediction"] == "YES" else "0\n")
    
    print(f"5. Output Files:")
    print(f"   Detailed results: {output_file}")
    print(f"   Predictions file: {pred_file}")
    print(f"   Test data: {demo_file}")
    print()
    
    print("6. Example Chain-of-Thought Response:")
    print("-" * 40)
    print(responses[0][:300] + "...")
    print()
    
    print("7. To run with real model:")
    print("   python run_qwen_480b.py --data_path demo_test_data.jsonl --output_folder ./demo_output")
    print()
    
    print("8. To calculate VD-Score:")
    print(f"   python ../calc_vd_score.py --pred_file {pred_file} --test_file {demo_file}")
    print()
    
    # Cleanup
    try:
        os.remove(demo_file)
        os.remove(output_file) 
        os.remove(pred_file)
        print("Demo files cleaned up.")
    except:
        pass
    
    print("Demo completed successfully! ✓")


if __name__ == "__main__":
    run_demo()
