"""Time limit management for agent execution"""

import time
from datetime import datetime, timedelta
from typing import Optional


class Clock:
    """Manages execution time limits"""
    
    def __init__(self, time_limit_minutes: int):
        """Initialize clock with time limit"""
        self.time_limit_minutes = time_limit_minutes
        self.time_limit_seconds = time_limit_minutes * 60
        self.start_time: Optional[float] = None
        self.end_time: Optional[datetime] = None
        
    def start(self):
        """Start the clock"""
        self.start_time = time.time()
        self.end_time = datetime.now() + timedelta(minutes=self.time_limit_minutes)
        print(f"⏰ Clock started at {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Time limit: {self.time_limit_minutes} minutes")
        print(f"   Will end at: {self.end_time.strftime('%H:%M:%S')}")
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def elapsed_minutes(self) -> float:
        """Get elapsed time in minutes"""
        return self.elapsed() / 60
    
    def remaining(self) -> float:
        """Get remaining time in seconds"""
        if self.start_time is None:
            return self.time_limit_seconds
        
        elapsed = self.elapsed()
        remaining = self.time_limit_seconds - elapsed
        return max(0, remaining)
    
    def remaining_minutes(self) -> float:
        """Get remaining time in minutes"""
        return self.remaining() / 60
    
    def should_stop(self) -> bool:
        """Check if time limit has been exceeded"""
        if self.start_time is None:
            return False
        
        return self.elapsed() >= self.time_limit_seconds
    
    def checkpoint(self, message: str = ""):
        """Print current time status"""
        if self.start_time is None:
            print("⏰ Clock not started")
            return
        
        elapsed = self.elapsed_minutes()
        remaining = self.remaining_minutes()
        
        if message:
            print(f"⏰ [{message}]", end=" ")
        else:
            print("⏰", end=" ")
        
        print(f"Elapsed: {elapsed:.1f} min, Remaining: {remaining:.1f} min")
        
        if remaining < 5:
            print("   ⚠️ WARNING: Less than 5 minutes remaining!")
        
        if self.should_stop():
            print("   🛑 TIME LIMIT EXCEEDED!")
    
    def get_progress(self) -> float:
        """Get progress as percentage (0-100)"""
        if self.start_time is None:
            return 0.0
        
        elapsed = self.elapsed()
        progress = (elapsed / self.time_limit_seconds) * 100
        return min(100, progress)
    
    def format_time(self, seconds: float) -> str:
        """Format seconds into readable time string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def summary(self) -> str:
        """Get summary of time usage"""
        if self.start_time is None:
            return "Clock not started"
        
        elapsed = self.elapsed()
        remaining = self.remaining()
        progress = self.get_progress()
        
        summary = f"Time Summary:\n"
        summary += f"  Started: {datetime.fromtimestamp(self.start_time).strftime('%H:%M:%S')}\n"
        summary += f"  Elapsed: {self.format_time(elapsed)}\n"
        summary += f"  Remaining: {self.format_time(remaining)}\n"
        summary += f"  Progress: {progress:.1f}%\n"
        
        if self.should_stop():
            summary += f"  Status: TIME LIMIT EXCEEDED\n"
        elif remaining < 300:  # Less than 5 minutes
            summary += f"  Status: ⚠️ CRITICAL - Less than 5 minutes!\n"
        elif remaining < 600:  # Less than 10 minutes
            summary += f"  Status: ⚠️ WARNING - Less than 10 minutes\n"
        else:
            summary += f"  Status: ✅ OK\n"
        
        return summary
