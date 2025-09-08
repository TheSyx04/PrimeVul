"""
Utility functions and prompts for Qwen vulnerability detection.
This file contains prompts adapted for Qwen models with enhanced CoT reasoning.
"""

# **System Instructions for Qwen**
QWEN_SYS_INST = """You are an expert security analyst specializing in static code analysis and vulnerability detection. Your task is to analyze C/C++ code for security vulnerabilities with high accuracy."""

# **Standard Classification Prompt for Qwen**
QWEN_PROMPT_INST = """Analyze the following C/C++ code for security vulnerabilities:

```c
{func}
```

Based on your security analysis, respond with EXACTLY one of these options:
- YES: If you detect any security vulnerability
- NO: If no security vulnerability is found

Provide only the option (YES or NO) without additional explanation."""

# **Enhanced Chain of Thought Prompt for Qwen**
QWEN_PROMPT_INST_COT = """Analyze the following C/C++ code for security vulnerabilities:

```c
{func}
```

Please perform a thorough security analysis following these steps:

1. **Code Understanding**: Briefly describe what this function does
2. **Input Analysis**: Identify all inputs and their sources (parameters, global variables, etc.)
3. **Data Flow**: Trace how data flows through the function
4. **Security Checks**: Look for common vulnerability patterns:
   - Buffer overflows/underflows
   - Memory leaks or use-after-free
   - Integer overflows/underflows
   - Null pointer dereferences
   - Format string vulnerabilities
   - SQL injection (if applicable)
   - Path traversal vulnerabilities
   - Race conditions
   - Unvalidated inputs
5. **Bounds Checking**: Verify if array/buffer accesses are properly bounded
6. **Memory Management**: Check for proper allocation/deallocation
7. **Input Validation**: Assess input sanitization and validation

After your analysis, conclude with EXACTLY one of these options:
- YES: If you detect any security vulnerability
- NO: If no security vulnerability is found

Make sure your final answer (YES or NO) is clearly stated at the end."""

# **Few-shot Examples for Qwen**

QWEN_ONESHOT_USER = """Analyze the following C/C++ code for security vulnerabilities:

```c
static char *clean_path(char *path)
{
        char *ch;
        char *ch2;
        char *str;
        str = xmalloc(strlen(path) + 1);
        ch = path;
        ch2 = str;
        while (true) {
                *ch2 = *ch;
                ch++;
                ch2++;
                if (!*(ch-1))
                        break;
                while (*(ch - 1) == '/' && *ch == '/')
                        ch++;
        }
        /* get rid of trailing / characters */
        while ((ch = strrchr(str, '/'))) {
                if (ch == str)
                        break;
                if (!*(ch+1))
                        *ch = 0;
                else
                        break;
        }
        return str;
}
```

Based on your security analysis, respond with EXACTLY one of these options:
- YES: If you detect any security vulnerability
- NO: If no security vulnerability is found

Provide only the option (YES or NO) without additional explanation."""

QWEN_ONESHOT_ASSISTANT = "NO"

QWEN_TWOSHOT_USER = """Analyze the following C/C++ code for security vulnerabilities:

```c
int64 ClientUsageTracker::GetCachedHostUsage(const std::string& host) {
   HostUsageMap::const_iterator found = cached_usage_.find(host);
   if (found == cached_usage_.end())
     return 0;

  int64 usage = 0;
  const UsageMap& map = found->second;
  for (UsageMap::const_iterator iter = map.begin();
       iter != map.end(); ++iter) {
    usage += iter->second;
  }
  return usage;
}
```

Based on your security analysis, respond with EXACTLY one of these options:
- YES: If you detect any security vulnerability
- NO: If no security vulnerability is found

Provide only the option (YES or NO) without additional explanation."""

QWEN_TWOSHOT_ASSISTANT = "YES"

# **Chain of Thought Few-shot Examples**

QWEN_COT_ONESHOT_USER = """Analyze the following C/C++ code for security vulnerabilities:

```c
void copy_data(char *dest, char *src, int len) {
    int i;
    for (i = 0; i < len; i++) {
        dest[i] = src[i];
    }
}
```

Please perform a thorough security analysis following these steps:

1. **Code Understanding**: Briefly describe what this function does
2. **Input Analysis**: Identify all inputs and their sources
3. **Data Flow**: Trace how data flows through the function
4. **Security Checks**: Look for common vulnerability patterns
5. **Bounds Checking**: Verify if array/buffer accesses are properly bounded
6. **Memory Management**: Check for proper allocation/deallocation
7. **Input Validation**: Assess input sanitization and validation

After your analysis, conclude with EXACTLY one of these options:
- YES: If you detect any security vulnerability
- NO: If no security vulnerability is found"""

QWEN_COT_ONESHOT_ASSISTANT = """1. **Code Understanding**: This function copies data from a source buffer to a destination buffer for a specified length.

2. **Input Analysis**: The function takes three inputs:
   - `dest`: destination buffer pointer
   - `src`: source buffer pointer  
   - `len`: number of bytes to copy

3. **Data Flow**: Data flows from src[i] to dest[i] in a loop from i=0 to i=len-1.

4. **Security Checks**: 
   - No bounds checking on destination buffer
   - No validation of buffer sizes
   - No null pointer checks
   - Buffer overflow vulnerability present

5. **Bounds Checking**: CRITICAL ISSUE - No verification that dest buffer can hold len bytes or that src buffer has len bytes available.

6. **Memory Management**: No memory allocation/deallocation in this function, but relies on caller to provide valid pointers.

7. **Input Validation**: No input validation - function trusts all parameters without verification.

**Final Answer**: YES"""

QWEN_COT_TWOSHOT_USER = """Analyze the following C/C++ code for security vulnerabilities:

```c
int safe_strlen(const char *str) {
    if (str == NULL) {
        return 0;
    }
    
    int len = 0;
    while (str[len] != '\\0' && len < 1000) {
        len++;
    }
    return len;
}
```

Please perform a thorough security analysis following these steps:

1. **Code Understanding**: Briefly describe what this function does
2. **Input Analysis**: Identify all inputs and their sources
3. **Data Flow**: Trace how data flows through the function
4. **Security Checks**: Look for common vulnerability patterns
5. **Bounds Checking**: Verify if array/buffer accesses are properly bounded
6. **Memory Management**: Check for proper allocation/deallocation
7. **Input Validation**: Assess input sanitization and validation

After your analysis, conclude with EXACTLY one of these options:
- YES: If you detect any security vulnerability
- NO: If no security vulnerability is found"""

QWEN_COT_TWOSHOT_ASSISTANT = """1. **Code Understanding**: This function calculates the length of a null-terminated string with safety bounds.

2. **Input Analysis**: Single input parameter:
   - `str`: const char pointer to null-terminated string

3. **Data Flow**: Function checks for null pointer, then iterates through string characters until null terminator or length limit is reached.

4. **Security Checks**: 
   - Null pointer validation present
   - Maximum length boundary (1000) prevents infinite loops
   - No buffer overrun possible due to bounds checking

5. **Bounds Checking**: Proper bounds checking with maximum length limit of 1000 characters.

6. **Memory Management**: No memory allocation/deallocation, function only reads data.

7. **Input Validation**: Good input validation with null pointer check and length limiting.

**Final Answer**: NO"""


def get_qwen_prompts():
    """
    Returns the appropriate prompt templates for Qwen model.
    """
    return {
        'system': QWEN_SYS_INST,
        'std_cls': QWEN_PROMPT_INST,
        'cot': QWEN_PROMPT_INST_COT,
        'oneshot_user': QWEN_ONESHOT_USER,
        'oneshot_assistant': QWEN_ONESHOT_ASSISTANT,
        'twoshot_user': QWEN_TWOSHOT_USER,
        'twoshot_assistant': QWEN_TWOSHOT_ASSISTANT,
        'cot_oneshot_user': QWEN_COT_ONESHOT_USER,
        'cot_oneshot_assistant': QWEN_COT_ONESHOT_ASSISTANT,
        'cot_twoshot_user': QWEN_COT_TWOSHOT_USER,
        'cot_twoshot_assistant': QWEN_COT_TWOSHOT_ASSISTANT
    }


def format_qwen_messages(prompt_data, strategy="cot", fewshot=True):
    """
    Format messages specifically for Qwen models with proper few-shot examples.
    
    Args:
        prompt_data: Dict containing the prompt information
        strategy: "std_cls" or "cot"
        fewshot: Whether to include few-shot examples
    
    Returns:
        List of formatted messages
    """
    prompts = get_qwen_prompts()
    
    messages = [{"role": "system", "content": prompts['system']}]
    
    if fewshot:
        if strategy == "cot":
            # Use CoT few-shot examples
            messages.extend([
                {"role": "user", "content": prompts['cot_oneshot_user']},
                {"role": "assistant", "content": prompts['cot_oneshot_assistant']},
                {"role": "user", "content": prompts['cot_twoshot_user']},
                {"role": "assistant", "content": prompts['cot_twoshot_assistant']}
            ])
        else:
            # Use standard few-shot examples
            messages.extend([
                {"role": "user", "content": prompts['oneshot_user']},
                {"role": "assistant", "content": prompts['oneshot_assistant']},
                {"role": "user", "content": prompts['twoshot_user']},
                {"role": "assistant", "content": prompts['twoshot_assistant']}
            ])
    
    # Add the actual prompt
    prompt_template = prompts[strategy]
    formatted_prompt = prompt_template.format(func=prompt_data["func"])
    messages.append({"role": "user", "content": formatted_prompt})
    
    return messages


def extract_qwen_prediction(response):
    """
    Enhanced prediction extraction for Qwen model responses.
    
    Args:
        response: Generated response text
        
    Returns:
        Extracted prediction: "YES", "NO", or "UNCLEAR"
    """
    if not response:
        return "UNCLEAR"
    
    response_clean = response.strip().upper()
    
    # Look for explicit final answers first
    final_answer_patterns = [
        "FINAL ANSWER: YES", "FINAL ANSWER: NO",
        "**FINAL ANSWER**: YES", "**FINAL ANSWER**: NO",
        "CONCLUSION: YES", "CONCLUSION: NO"
    ]
    
    for pattern in final_answer_patterns:
        if pattern in response_clean:
            return "YES" if "YES" in pattern else "NO"
    
    # Look for standalone YES/NO at the end
    lines = response_clean.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line == "YES":
            return "YES"
        elif line == "NO":
            return "NO"
    
    # Look for YES/NO with minimal context
    if response_clean.endswith(" YES") or response_clean.endswith(": YES"):
        return "YES"
    elif response_clean.endswith(" NO") or response_clean.endswith(": NO"):
        return "NO"
    
    # Count vulnerability indicators
    vulnerability_indicators = [
        "VULNERABILITY", "VULNERABLE", "SECURITY ISSUE", "SECURITY FLAW",
        "BUFFER OVERFLOW", "MEMORY LEAK", "NULL POINTER", "INTEGER OVERFLOW",
        "USE AFTER FREE", "DOUBLE FREE", "FORMAT STRING"
    ]
    
    safe_indicators = [
        "NO VULNERABILITY", "NOT VULNERABLE", "SECURE", "SAFE",
        "NO SECURITY ISSUE", "NO MEMORY ISSUES", "PROPERLY VALIDATED"
    ]
    
    vuln_count = sum(1 for indicator in vulnerability_indicators if indicator in response_clean)
    safe_count = sum(1 for indicator in safe_indicators if indicator in response_clean)
    
    if safe_count > vuln_count:
        return "NO"
    elif vuln_count > safe_count:
        return "YES"
    
    return "UNCLEAR"
