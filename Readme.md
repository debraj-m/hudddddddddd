# Multi-Modal Classification for Huddler - Implementation Guide

## What's the Multi-Modal Approach?

Think of it like having multiple specialized "experts" working together to make a decision. Instead of one big AI trying to solve everything, we break the problem into smaller, focused components:

1. **Question Detector**: "Is this even a question?"
2. **Answerability Classifier**: "Can this be answered factually?"
3. **Direction Detector**: "Is this question meant for me?"
4. **Context Analyzer**: "What kind of meeting is this?"
5. **Confidence Calculator**: "How sure are we about our decision?"

Each component is like a specialist doctor - one focuses on detecting questions, another on understanding context, etc. Then we combine their "opinions" to make the final decision.

## Why This Works Better

**Traditional Approach**: One big model tries to learn everything
- Hard to debug when it fails
- Needs tons of training data
- Black box decisions

**Multi-Modal Approach**: Multiple specialized components
- Each component is simple and focused
- Can start with rules, add ML gradually
- Easy to understand why it made a decision
- Fix problems in specific areas without breaking everything

## Architecture Overview

```
Input Message → [Question Detector] → [Answerability Classifier] → [Direction Detector] → [Confidence Calculator] → Final Decision
                       ↓                         ↓                        ↓                      ↑
                [Context Analyzer] ←→ [Speaker Pattern Learner] ←→ [Conversation History] ←→ [User Profile]
```

## Component 1: Question Detector

This component figures out if the input text contains actual questions.

```python
import re
import spacy
from typing import List, Tuple

class QuestionDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.question_patterns = [
            r'\b(what|how|when|where|why|who|which|can|could|would|should|do|does|did|is|are|was|were)\b.*\?',
            r'.*\?\s*$',  # Ends with question mark
            r'\b(tell me|explain|describe|clarify)\b.*',  # Imperative questions
        ]
        
    def detect_questions(self, text: str) -> List[str]:
        """Extract potential questions from text"""
        # Split by question marks first
        potential_questions = []
        
        # Method 1: Split by question marks
        for segment in text.split('?'):
            if segment.strip():
                question = segment.strip() + '?'
                if self._is_question(question):
                    potential_questions.append(question)
        
        # Method 2: Detect imperative questions without question marks
        doc = self.nlp(text)
        for sent in doc.sents:
            if self._is_imperative_question(sent.text):
                potential_questions.append(sent.text)
        
        return list(set(potential_questions))  # Remove duplicates
    
    def _is_question(self, text: str) -> bool:
        """Check if text looks like a question"""
        if len(text.strip()) < 5:
            return False
            
        for pattern in self.question_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _is_imperative_question(self, text: str) -> bool:
        """Detect imperative questions like 'Tell me about...'"""
        imperative_starters = ['tell me', 'explain', 'describe', 'walk me through', 'show me']
        text_lower = text.lower()
        return any(text_lower.startswith(starter) for starter in imperative_starters)
```

## Component 2: Answerability Classifier

This determines if a question can actually be answered vs being rhetorical/speculative.

```python
from enum import Enum
from dataclasses import dataclass

class QuestionType(Enum):
    FACTUAL = "factual"           # "What was our revenue?"
    OPINION = "opinion"           # "What do you think?"
    RHETORICAL = "rhetorical"     # "Isn't it nice?"
    SPECULATIVE = "speculative"   # "What if aliens exist?"
    CLARIFICATION = "clarification"  # "Can you explain that?"

@dataclass
class AnswerabilityResult:
    question_type: QuestionType
    answerable: bool
    confidence: float
    reasoning: str

class AnswerabilityClassifier:
    def __init__(self):
        self.patterns = {
            'rhetorical': [
                r"don't you think", r"isn't it", r"wouldn't you say", 
                r"right\?", r"you know\?", r"isn't that"
            ],
            'speculative': [
                r"what if", r"imagine if", r"suppose", r"hypothetically",
                r"in theory", r"theoretically"
            ],
            'opinion': [
                r"what do you think", r"your opinion", r"how do you feel",
                r"what's your take", r"do you believe"
            ],
            'factual': [
                r"what is", r"what was", r"what were", r"how many", 
                r"when did", r"where is", r"who is"
            ],
            'clarification': [
                r"can you explain", r"clarify", r"elaborate", 
                r"tell me more", r"walk me through"
            ]
        }
    
    def classify(self, question: str, user_context: str) -> AnswerabilityResult:
        """Classify question answerability"""
        question_lower = question.lower()
        
        # Check each pattern type
        for question_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return self._create_result(question_type, question, user_context)
        
        # Default classification
        if any(marker in question_lower for marker in ['what', 'how', 'when', 'where', 'why']):
            return self._create_result('factual', question, user_context)
        
        return AnswerabilityResult(
            QuestionType.RHETORICAL, False, 0.3, "No clear pattern detected"
        )
    
    def _create_result(self, question_type: str, question: str, user_context: str) -> AnswerabilityResult:
        """Create classification result based on type"""
        type_enum = QuestionType(question_type)
        
        if question_type == 'rhetorical':
            return AnswerabilityResult(type_enum, False, 0.9, "Rhetorical question pattern detected")
        elif question_type == 'speculative':
            return AnswerabilityResult(type_enum, False, 0.8, "Speculative question - no definitive answer")
        elif question_type in ['factual', 'opinion', 'clarification']:
            # Check if user can answer based on context
            can_answer = self._can_user_answer(question, user_context)
            confidence = 0.8 if can_answer else 0.4
            reasoning = f"{question_type.title()} question - user {'can' if can_answer else 'might not'} answer"
            return AnswerabilityResult(type_enum, can_answer, confidence, reasoning)
        
        return AnswerabilityResult(type_enum, False, 0.3, "Unknown pattern")
    
    def _can_user_answer(self, question: str, user_context: str) -> bool:
        """Check if user has relevant knowledge to answer"""
        question_lower = question.lower()
        context_lower = user_context.lower()
        
        # Extract key terms from question
        tech_terms = ['database', 'sql', 'nosql', 'react', 'python', 'javascript', 
                     'system', 'design', 'api', 'backend', 'frontend']
        
        business_terms = ['revenue', 'sales', 'strategy', 'market', 'customer']
        
        # Check if question is about something in user's expertise
        if any(term in question_lower for term in tech_terms):
            return any(term in context_lower for term in tech_terms)
        
        if any(term in question_lower for term in business_terms):
            return 'business' in context_lower or 'manager' in context_lower
        
        return True  # Default to true for general questions
```

## Component 3: Direction Detector

This figures out if the question is meant for the user specifically.

```python
from enum import Enum

class AddressingType(Enum):
    DIRECT_NAME = "direct_name"      # "Sarah, can you..."
    ROLE_BASED = "role_based"        # "Can the developer..."
    CONTEXTUAL = "contextual"        # "Can you..." (context dependent)
    GENERAL = "general"              # "Can anyone..."
    NONE = "none"                    # No clear addressing

@dataclass
class DirectionResult:
    addressing_type: AddressingType
    directed_at_user: bool
    confidence: float
    reasoning: str

class DirectionDetector:
    def __init__(self, user_name: str, user_role: str):
        self.user_name = user_name.lower()
        self.user_role = user_role.lower()
        self.conversation_history = []
        
    def detect_direction(self, question: str, speaker_id: int, conversation_context: List) -> DirectionResult:
        """Detect if question is directed at user"""
        question_lower = question.lower()
        
        # Check for direct name addressing
        if self.user_name in question_lower:
            # Look for pattern: "Name, question" or "question, Name"
            name_patterns = [
                rf'\b{self.user_name}\b[,:]?\s*(can|could|would|will|do)',
                rf'(can|could|would|will|do).*\b{self.user_name}\b'
            ]
            for pattern in name_patterns:
                if re.search(pattern, question_lower):
                    return DirectionResult(
                        AddressingType.DIRECT_NAME, True, 0.95,
                        f"Direct name addressing: {self.user_name}"
                    )
        
        # Check for role-based addressing
        role_patterns = [
            rf'\b{self.user_role}\b',
            r'\b(developer|engineer|programmer|analyst)\b'
        ]
        for pattern in role_patterns:
            if re.search(pattern, question_lower):
                return DirectionResult(
                    AddressingType.ROLE_BASED, True, 0.85,
                    f"Role-based addressing detected"
                )
        
        # Check for general addressing (not for user)
        general_patterns = [
            r'\b(anyone|everyone|somebody|anybody)\b',
            r'\b(does anyone|can anyone|would anyone)\b'
        ]
        for pattern in general_patterns:
            if re.search(pattern, question_lower):
                return DirectionResult(
                    AddressingType.GENERAL, False, 0.9,
                    "General question for anyone"
                )
        
        # Check for contextual addressing ("you", "your")
        contextual_patterns = [r'\b(you|your)\b']
        for pattern in contextual_patterns:
            if re.search(pattern, question_lower):
                confidence = self._analyze_contextual_confidence(question, speaker_id, conversation_context)
                return DirectionResult(
                    AddressingType.CONTEXTUAL, confidence > 0.6, confidence,
                    f"Contextual addressing with {confidence:.2f} confidence"
                )
        
        return DirectionResult(
            AddressingType.NONE, False, 0.2,
            "No clear addressing pattern"
        )
    
    def _analyze_contextual_confidence(self, question: str, speaker_id: int, context: List) -> float:
        """Analyze contextual clues for 'you' questions"""
        confidence = 0.5  # Base confidence
        
        # If user spoke recently, next question likely for them
        if context and context[-1].get('role') == 'user':
            confidence += 0.3
        
        # Check if this speaker often asks user questions
        speaker_history = [msg for msg in context if msg.get('speaker') == speaker_id]
        if len(speaker_history) > 2:
            user_directed_count = sum(1 for msg in speaker_history 
                                    if 'user_directed' in msg and msg['user_directed'])
            if user_directed_count / len(speaker_history) > 0.6:
                confidence += 0.2
        
        return min(confidence, 1.0)
```

## Component 4: Context Analyzer

This understands the meeting type and adjusts behavior accordingly.

```python
class ContextAnalyzer:
    def __init__(self):
        self.meeting_types = {
            'interview': {
                'keywords': ['interview', 'position', 'candidate', 'hiring'],
                'question_likelihood': 0.9,  # Questions usually for interviewee
                'response_expected': True
            },
            'presentation': {
                'keywords': ['presentation', 'demo', 'show', 'explain'],
                'question_likelihood': 0.7,
                'response_expected': True
            },
            'status_meeting': {
                'keywords': ['status', 'update', 'progress', 'standup'],
                'question_likelihood': 0.6,
                'response_expected': True
            },
            'brainstorm': {
                'keywords': ['brainstorm', 'ideas', 'think', 'creative'],
                'question_likelihood': 0.8,
                'response_expected': True
            }
        }
    
    def analyze_context(self, meeting_context: str, conversation_history: List) -> dict:
        """Analyze meeting context to adjust question handling"""
        context_lower = meeting_context.lower()
        
        # Determine meeting type
        meeting_type = 'general'
        for mtype, config in self.meeting_types.items():
            if any(keyword in context_lower for keyword in config['keywords']):
                meeting_type = mtype
                break
        
        # Analyze conversation patterns
        total_messages = len(conversation_history)
        user_messages = sum(1 for msg in conversation_history if msg.get('role') == 'user')
        
        participation_ratio = user_messages / total_messages if total_messages > 0 else 0
        
        return {
            'meeting_type': meeting_type,
            'config': self.meeting_types.get(meeting_type, self.meeting_types['general']),
            'participation_ratio': participation_ratio,
            'user_engagement': 'high' if participation_ratio > 0.3 else 'medium' if participation_ratio > 0.1 else 'low'
        }
```

## Main Pipeline Integration

Now we put all components together into one decision-making system.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class HuddlerDecision:
    should_respond: bool
    confidence: float
    question: str
    reasoning: str
    suggested_response: Optional[str] = None

class HuddlerMultiModalPipeline:
    def __init__(self, user_name: str, user_role: str, user_context: str, meeting_context: str):
        self.user_name = user_name
        self.user_context = user_context
        self.meeting_context = meeting_context
        
        # Initialize components
        self.question_detector = QuestionDetector()
        self.answerability_classifier = AnswerabilityClassifier()
        self.direction_detector = DirectionDetector(user_name, user_role)
        self.context_analyzer = ContextAnalyzer()
        
        self.conversation_history = []
        
    def process_message(self, message: dict) -> Optional[List[HuddlerDecision]]:
        """Main processing pipeline"""
        # Only process third-party messages
        if message.get('role') != 'third_party':
            self.conversation_history.append(message)
            return None
        
        text = message.get('text', '')
        speaker_id = message.get('speaker')
        
        # Step 1: Detect questions
        questions = self.question_detector.detect_questions(text)
        if not questions:
            self.conversation_history.append(message)
            return None
        
        # Step 2: Analyze context
        context_analysis = self.context_analyzer.analyze_context(
            self.meeting_context, self.conversation_history
        )
        
        decisions = []
        for question in questions:
            # Step 3: Classify answerability
            answerability = self.answerability_classifier.classify(question, self.user_context)
            
            # Step 4: Detect direction
            direction = self.direction_detector.detect_direction(
                question, speaker_id, self.conversation_history
            )
            
            # Step 5: Calculate final confidence
            final_decision = self._make_final_decision(
                question, answerability, direction, context_analysis
            )
            
            if final_decision.should_respond:
                decisions.append(final_decision)
        
        self.conversation_history.append(message)
        return decisions if decisions else None
    
    def _make_final_decision(self, question: str, answerability: AnswerabilityResult, 
                           direction: DirectionResult, context: dict) -> HuddlerDecision:
        """Combine all component results into final decision"""
        
        # Base decision logic
        should_respond = answerability.answerable and direction.directed_at_user
        
        # Calculate combined confidence
        base_confidence = min(answerability.confidence, direction.confidence)
        
        # Adjust confidence based on context
        meeting_config = context['config']
        if should_respond:
            # Boost confidence based on meeting type
            context_boost = meeting_config['question_likelihood'] * 0.2
            final_confidence = min(base_confidence + context_boost, 1.0)
        else:
            final_confidence = base_confidence * 0.5  # Lower confidence for negative decisions
        
        # Generate reasoning
        reasoning_parts = [
            f"Question type: {answerability.question_type.value}",
            f"Addressing: {direction.addressing_type.value}",
            f"Meeting type: {context['meeting_type']}"
        ]
        reasoning = " | ".join(reasoning_parts)
        
        # Generate suggested response if should respond
        suggested_response = None
        if should_respond and final_confidence > 0.7:
            suggested_response = self._generate_response_suggestion(question, answerability.question_type)
        
        return HuddlerDecision(
            should_respond=should_respond and final_confidence > 0.6,
            confidence=final_confidence,
            question=question,
            reasoning=reasoning,
            suggested_response=suggested_response
        )
    
    def _generate_response_suggestion(self, question: str, question_type: QuestionType) -> str:
        """Generate simple response suggestions"""
        if question_type == QuestionType.FACTUAL:
            if 'experience' in question.lower():
                return "Based on my experience in..."
            elif 'technology' in question.lower() or 'skill' in question.lower():
                return "I have worked with..."
            else:
                return "Let me answer that..."
        
        elif question_type == QuestionType.OPINION:
            return "I think..."
        
        elif question_type == QuestionType.CLARIFICATION:
            return "Sure, let me explain..."
        
        return "That's a great question..."

# Example Usage
def main():
    # Initialize the pipeline
    pipeline = HuddlerMultiModalPipeline(
        user_name="Sarah",
        user_role="software engineer", 
        user_context="Sarah is a 28-year-old software engineer with 5 years of experience...",
        meeting_context="Technical job interview for Senior Developer position..."
    )
    
    # Test messages
    test_messages = [
        {
            "role": "third_party",
            "speaker": 0,
            "text": "Sarah, can you explain the difference between SQL and NoSQL databases?"
        },
        {
            "role": "third_party", 
            "speaker": 0,
            "text": "Don't you think the weather is nice today?"
        },
        {
            "role": "third_party",
            "speaker": 0, 
            "text": "Does anyone know where the bathroom is?"
        }
    ]
    
    for message in test_messages:
        decisions = pipeline.process_message(message)
        if decisions:
            for decision in decisions:
                print(f"\nMessage: {message['text']}")
                print(f"Should respond: {decision.should_respond}")
                print(f"Confidence: {decision.confidence:.2f}")
                print(f"Question: {decision.question}")
                print(f"Reasoning: {decision.reasoning}")
                if decision.suggested_response:
                    print(f"Suggested: {decision.suggested_response}")
        else:
            print(f"\nMessage: {message['text']} -> No response needed")

if __name__ == "__main__":
    main()
```

## Why This Approach Works

**1. Modularity**: Each component handles one specific task well. If question detection fails, you can fix just that component without touching the others.

**2. Transparency**: You always know WHY the system made a decision. "Confidence low because question was rhetorical + not directly addressed to user."

**3. Gradual Improvement**: Start with simple rules, then upgrade individual components to use machine learning as you get more data.

**4. Real-world Ready**: Works with actual meeting scenarios right away, doesn't need months of training data.

**5. Customizable**: Easy to adjust thresholds and add new patterns for different meeting types or user preferences.

The key insight is that breaking a complex problem into simpler, focused sub-problems makes the whole system more reliable, debuggable, and maintainable.
