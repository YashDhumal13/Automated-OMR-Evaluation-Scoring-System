"""
Scoring Module

This module handles the comparison of student answers with the correct answer key
and calculates comprehensive scoring results including total scores and subject-wise breakdown.

Key Features:
- Answer key loading from Excel files
- Flexible scoring schemes (positive/negative marking)
- Subject-wise score calculation
- Detailed performance analytics
- Support for multiple answer formats
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for scoring system."""
    correct_points: float = 1.0
    incorrect_points: float = 0.0
    unanswered_points: float = 0.0
    multiple_answer_points: float = 0.0  # Points for multiple answers (usually 0 or negative)


@dataclass
class QuestionScore:
    """Score details for a single question."""
    question_number: int
    student_answer: Optional[str]
    correct_answer: str
    is_correct: bool
    points_earned: float
    subject: Optional[str] = None


@dataclass
class ScoreResult:
    """Complete scoring result for a student."""
    student_id: Optional[str]
    total_score: float
    total_possible: float
    percentage: float
    correct_answers: int
    incorrect_answers: int
    unanswered_questions: int
    multiple_answers: int
    question_scores: List[QuestionScore]
    subject_scores: Dict[str, Dict[str, float]]  # subject -> {score, possible, percentage}


class ScoreCalculator:
    """
    Calculates scores by comparing student answers with the answer key.
    
    This class handles loading answer keys, comparing answers, and generating
    comprehensive scoring reports.
    """
    
    def __init__(self, scoring_config: Optional[ScoringConfig] = None):
        """
        Initialize the score calculator.
        
        Args:
            scoring_config: Configuration for scoring scheme
        """
        self.scoring_config = scoring_config or ScoringConfig()
        self.answer_key = {}
        self.subject_mapping = {}
    
    def load_answer_key_from_excel(self, file_path: str, sheet_name: Optional[str] = None) -> bool:
        """
        Load answer key from an Excel file.
        
        Args:
            file_path: Path to the Excel file containing the answer key
            sheet_name: Name of the sheet to read (if None, reads first sheet)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read Excel file
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            
            logger.info(f"Loaded answer key with columns: {df.columns.tolist()}")
            
            # Parse answer key based on expected format
            self.answer_key, self.subject_mapping = self._parse_answer_key_dataframe(df)
            
            logger.info(f"Loaded {len(self.answer_key)} questions from answer key")
            return True
            
        except Exception as e:
            logger.error(f"Error loading answer key from {file_path}: {str(e)}")
            return False
    
    def _parse_answer_key_dataframe(self, df: pd.DataFrame) -> Tuple[Dict[int, str], Dict[int, str]]:
        """
        Parse the answer key dataframe into usable dictionaries.
        
        Args:
            df: DataFrame containing the answer key
            
        Returns:
            Tuple of (answer_key_dict, subject_mapping_dict)
        """
        answer_key = {}
        subject_mapping = {}
        
        # Try to identify columns automatically
        question_col = None
        answer_col = None
        subject_col = None
        
        # Look for question number column
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['question', 'q', 'no', 'number']):
                question_col = col
                break
        
        # Look for answer column
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['answer', 'key', 'correct', 'solution']):
                answer_col = col
                break
        
        # Look for subject column
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['subject', 'topic', 'section', 'category']):
                subject_col = col
                break
        
        # If automatic detection fails, use positional approach
        if question_col is None and len(df.columns) >= 2:
            question_col = df.columns[0]
        if answer_col is None and len(df.columns) >= 2:
            answer_col = df.columns[1]
        if subject_col is None and len(df.columns) >= 3:
            subject_col = df.columns[2]
        
        # Parse the data
        for index, row in df.iterrows():
            try:
                # Get question number
                if question_col:
                    q_num = int(row[question_col])
                else:
                    q_num = index + 1  # Use row index + 1 as question number
                
                # Get correct answer
                if answer_col:
                    answer = str(row[answer_col]).strip().upper()
                    if answer and answer != 'NAN':
                        answer_key[q_num] = answer
                
                # Get subject if available
                if subject_col and q_num in answer_key:
                    subject = str(row[subject_col]).strip()
                    if subject and subject != 'NAN':
                        subject_mapping[q_num] = subject
                        
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping row {index} due to parsing error: {e}")
                continue
        
        return answer_key, subject_mapping
    
    def set_answer_key_manually(self, answer_key: Dict[int, str], 
                               subject_mapping: Optional[Dict[int, str]] = None):
        """
        Set the answer key manually.
        
        Args:
            answer_key: Dictionary mapping question numbers to correct answers
            subject_mapping: Optional dictionary mapping question numbers to subjects
        """
        self.answer_key = {k: str(v).strip().upper() for k, v in answer_key.items()}
        self.subject_mapping = subject_mapping or {}
        logger.info(f"Manually set answer key with {len(self.answer_key)} questions")
    
    def calculate_score(self, student_answers: Dict[int, str], 
                       student_id: Optional[str] = None) -> ScoreResult:
        """
        Calculate the score for a student's answers.
        
        Args:
            student_answers: Dictionary mapping question numbers to student answers
            student_id: Optional student identifier
            
        Returns:
            ScoreResult object with comprehensive scoring information
        """
        if not self.answer_key:
            raise ValueError("Answer key not loaded. Please load an answer key first.")
        
        question_scores = []
        total_score = 0.0
        correct_count = 0
        incorrect_count = 0
        unanswered_count = 0
        multiple_answer_count = 0
        
        # Process each question in the answer key
        for question_num, correct_answer in self.answer_key.items():
            student_answer = student_answers.get(question_num)
            
            # Determine the score for this question
            points_earned, is_correct = self._score_single_question(
                student_answer, correct_answer
            )
            
            # Create question score object
            question_score = QuestionScore(
                question_number=question_num,
                student_answer=student_answer,
                correct_answer=correct_answer,
                is_correct=is_correct,
                points_earned=points_earned,
                subject=self.subject_mapping.get(question_num)
            )
            
            question_scores.append(question_score)
            total_score += points_earned
            
            # Update counters
            if student_answer is None:
                unanswered_count += 1
            elif ',' in str(student_answer):  # Multiple answers
                multiple_answer_count += 1
            elif is_correct:
                correct_count += 1
            else:
                incorrect_count += 1
        
        # Calculate totals
        total_possible = len(self.answer_key) * self.scoring_config.correct_points
        percentage = (total_score / total_possible * 100) if total_possible > 0 else 0
        
        # Calculate subject-wise scores
        subject_scores = self._calculate_subject_scores(question_scores)
        
        return ScoreResult(
            student_id=student_id,
            total_score=total_score,
            total_possible=total_possible,
            percentage=percentage,
            correct_answers=correct_count,
            incorrect_answers=incorrect_count,
            unanswered_questions=unanswered_count,
            multiple_answers=multiple_answer_count,
            question_scores=question_scores,
            subject_scores=subject_scores
        )
    
    def _score_single_question(self, student_answer: Optional[str], 
                              correct_answer: str) -> Tuple[float, bool]:
        """
        Score a single question.
        
        Args:
            student_answer: Student's answer (None if unanswered)
            correct_answer: Correct answer from the key
            
        Returns:
            Tuple of (points_earned, is_correct)
        """
        # Handle unanswered questions
        if student_answer is None:
            return self.scoring_config.unanswered_points, False
        
        # Normalize answers for comparison
        student_answer = str(student_answer).strip().upper()
        correct_answer = str(correct_answer).strip().upper()
        
        # Handle multiple answers (comma-separated)
        if ',' in student_answer:
            return self.scoring_config.multiple_answer_points, False
        
        # Check if answer is correct
        if student_answer == correct_answer:
            return self.scoring_config.correct_points, True
        else:
            return self.scoring_config.incorrect_points, False
    
    def _calculate_subject_scores(self, question_scores: List[QuestionScore]) -> Dict[str, Dict[str, float]]:
        """
        Calculate scores grouped by subject.
        
        Args:
            question_scores: List of individual question scores
            
        Returns:
            Dictionary mapping subjects to their score information
        """
        subject_scores = {}
        
        # Group questions by subject
        subject_questions = {}
        for q_score in question_scores:
            subject = q_score.subject or "General"
            if subject not in subject_questions:
                subject_questions[subject] = []
            subject_questions[subject].append(q_score)
        
        # Calculate scores for each subject
        for subject, questions in subject_questions.items():
            total_score = sum(q.points_earned for q in questions)
            total_possible = len(questions) * self.scoring_config.correct_points
            percentage = (total_score / total_possible * 100) if total_possible > 0 else 0
            correct_count = sum(1 for q in questions if q.is_correct)
            
            subject_scores[subject] = {
                'score': total_score,
                'possible': total_possible,
                'percentage': percentage,
                'correct_count': correct_count,
                'total_questions': len(questions)
            }
        
        return subject_scores
    
    def generate_detailed_report(self, score_result: ScoreResult) -> Dict[str, Any]:
        """
        Generate a detailed scoring report.
        
        Args:
            score_result: ScoreResult object
            
        Returns:
            Dictionary containing detailed report information
        """
        report = {
            'summary': {
                'student_id': score_result.student_id,
                'total_score': score_result.total_score,
                'total_possible': score_result.total_possible,
                'percentage': round(score_result.percentage, 2),
                'grade': self._calculate_grade(score_result.percentage)
            },
            'statistics': {
                'correct_answers': score_result.correct_answers,
                'incorrect_answers': score_result.incorrect_answers,
                'unanswered_questions': score_result.unanswered_questions,
                'multiple_answers': score_result.multiple_answers,
                'total_questions': len(score_result.question_scores)
            },
            'subject_breakdown': score_result.subject_scores,
            'question_details': []
        }
        
        # Add question-by-question details
        for q_score in score_result.question_scores:
            question_detail = {
                'question_number': q_score.question_number,
                'student_answer': q_score.student_answer,
                'correct_answer': q_score.correct_answer,
                'is_correct': q_score.is_correct,
                'points_earned': q_score.points_earned,
                'subject': q_score.subject
            }
            report['question_details'].append(question_detail)
        
        return report
    
    def _calculate_grade(self, percentage: float) -> str:
        """
        Calculate letter grade based on percentage.
        
        Args:
            percentage: Score percentage
            
        Returns:
            Letter grade string
        """
        if percentage >= 90:
            return 'A'
        elif percentage >= 80:
            return 'B'
        elif percentage >= 70:
            return 'C'
        elif percentage >= 60:
            return 'D'
        else:
            return 'F'
    
    def export_results_to_json(self, score_result: ScoreResult, file_path: str) -> bool:
        """
        Export scoring results to a JSON file.
        
        Args:
            score_result: ScoreResult object to export
            file_path: Path where to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            report = self.generate_detailed_report(score_result)
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Results exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results to {file_path}: {str(e)}")
            return False
    
    def batch_score_multiple_students(self, students_answers: Dict[str, Dict[int, str]]) -> Dict[str, ScoreResult]:
        """
        Score multiple students at once.
        
        Args:
            students_answers: Dictionary mapping student IDs to their answers
            
        Returns:
            Dictionary mapping student IDs to their ScoreResult objects
        """
        results = {}
        
        for student_id, answers in students_answers.items():
            try:
                results[student_id] = self.calculate_score(answers, student_id)
                logger.info(f"Scored student {student_id}: {results[student_id].percentage:.1f}%")
            except Exception as e:
                logger.error(f"Error scoring student {student_id}: {str(e)}")
        
        return results
    
    def get_answer_key_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded answer key.
        
        Returns:
            Dictionary with answer key information
        """
        if not self.answer_key:
            return {"loaded": False, "message": "No answer key loaded"}
        
        subjects = list(set(self.subject_mapping.values())) if self.subject_mapping else []
        
        return {
            "loaded": True,
            "total_questions": len(self.answer_key),
            "question_range": f"{min(self.answer_key.keys())}-{max(self.answer_key.keys())}",
            "subjects": subjects,
            "has_subject_mapping": bool(self.subject_mapping),
            "scoring_config": {
                "correct_points": self.scoring_config.correct_points,
                "incorrect_points": self.scoring_config.incorrect_points,
                "unanswered_points": self.scoring_config.unanswered_points,
                "multiple_answer_points": self.scoring_config.multiple_answer_points
            }
        }