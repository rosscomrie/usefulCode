SAMPLE_TEXT = """
Artificial Intelligence (AI) is transforming how we live and work. Machine learning, a subset of AI, 
enables computers to learn from data without explicit programming. Deep learning, a type of machine 
learning, uses neural networks with multiple layers to process complex patterns.

Natural Language Processing (NLP) is another crucial area of AI. It helps computers understand, 
interpret, and generate human language. Applications include machine translation, sentiment analysis, 
and chatbots.

Computer vision enables machines to understand and process visual information from the world. 
This technology is used in facial recognition, autonomous vehicles, and medical image analysis.

Robotics combines AI with mechanical engineering to create machines that can perform physical tasks. 
Modern robots use sensors, AI, and sophisticated control systems to interact with their environment.

Ethics in AI is an important consideration. Issues include bias in algorithms, privacy concerns, 
and the impact of automation on employment. Responsible AI development requires addressing these 
challenges while maximizing benefits to society.
"""

# A larger, more complex sample for thorough testing
COMPLEX_SAMPLE_TEXT = """
Introduction to Modern Computing Systems

Chapter 1: Fundamentals of Computing
Computing systems are the backbone of modern technology. They comprise various components working
in harmony to process information. The central processing unit (CPU) acts as the brain, while
memory systems store both instructions and data.

Section 1.1: Hardware Architecture
Modern processors utilize complex architectures including multiple cores and cache hierarchies.
The memory hierarchy includes registers, cache levels (L1, L2, L3), RAM, and secondary storage.
Understanding this architecture is crucial for optimal system performance.

Section 1.2: Software Systems
Operating systems manage hardware resources and provide services to applications. They handle
tasks such as process scheduling, memory management, and file system operations. Modern operating
systems support both single-user and multi-user environments.

Chapter 2: Networking and Communication
Computer networks enable information exchange between systems. Protocols govern how data is
transmitted, ensuring reliable communication across various network types.

Section 2.1: Network Protocols
The TCP/IP protocol suite forms the foundation of internet communication. It includes protocols
for different network layers, each serving specific purposes in data transmission.

Section 2.2: Security Considerations
Network security involves protecting systems and data from unauthorized access. Encryption,
authentication, and access control mechanisms help maintain system integrity and data
confidentiality.
"""

TEST_QUERIES = [
    "What is machine learning?",
    "Explain the relationship between AI and robotics.",
    "What are the ethical concerns in AI?",
    "How does computer vision work?",
    "What is the role of NLP in AI?",
    "Describe the basic components of a computing system",
    "How do modern processors handle multiple tasks?",
    "What are the key aspects of network security?",
    "Explain the importance of memory hierarchy",
    "What is the role of operating systems?"
]

def get_sample_text(text_type: str = 'basic') -> str:
    """Get sample text based on specified type."""
    if text_type == 'complex':
        return COMPLEX_SAMPLE_TEXT
    return SAMPLE_TEXT

def get_test_queries(category: str = None) -> list[str]:
    """Get test queries, optionally filtered by category."""
    if category == 'ai':
        return TEST_QUERIES[:5]
    elif category == 'systems':
        return TEST_QUERIES[5:]
    return TEST_QUERIES