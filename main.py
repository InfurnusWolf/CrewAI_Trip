import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import requests
# from langchain_openai import LLM
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()

@dataclass
class TripDetails:
    """
    Comprehensive trip details with multi-destination support
    """
    from_destination: str  # Origin city/country
    to_destination: str    # Destination city/country
    budget_min: str
    budget_max: str
    interests: List[str]
    travel_style: str
    group_size: int
    travel_dates: Dict[str, str]  # Start and end dates
    dietary_restrictions: Optional[List[str]] = None
    accessibility_needs: Optional[List[str]] = None
    
    def validate(self):
        """
        Validate trip details
        Raises ValueError for invalid inputs
        """
        if self.budget_min <= 0 or self.budget_max <= 0:
            raise ValueError("Budget must be positive")
        
        if self.budget_min > self.budget_max:
            raise ValueError("Minimum budget cannot exceed maximum budget")
        
        if self.group_size <= 0:
            raise ValueError("Group size must be positive")
        
        if self.from_destination == self.to_destination:
            raise ValueError("Origin and destination cannot be the same")


class MultiDestinationTripPlanner:
    def __init__(self, trip_details: TripDetails):
        """
        Initialize Multi-Destination Trip Planner
        
        :param trip_details: Comprehensive trip details
        """
        # Validate trip details
        trip_details.validate()
        self.trip_details = trip_details
        
        # Initialize language model
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.7
        )
        # self.llm=LLM(
        # model="ollama/llama3.1",
        # base_url="http://localhost:11434"
        # )
        
        # Create specialized agents
        self.route_analyzer = self._create_route_analyst()
        self.travel_logistics_agent = self._create_travel_logistics_agent()
        self.destination_researcher = self._create_destination_researcher()
        self.itinerary_planner = self._create_itinerary_planner()
        self.experience_curator = self._create_experience_curator()
    
    def _create_route_analyst(self):
        """Create an agent specialized in route and connection analysis"""
        return Agent(
            role="Global Route Strategist",
            goal=f"Analyze travel route from {self.trip_details.from_destination} "
                 f"to {self.trip_details.to_destination}, "
                 "identifying optimal travel paths and connections",
            backstory="An expert in global travel routing with deep knowledge of international connections",
            verbose=True,
            llm=self.llm
        )
    
    def _create_travel_logistics_agent(self):
        """Create an agent for travel logistics and transportation"""
        return Agent(
            role="Travel Logistics Coordinator",
            goal="Develop comprehensive travel logistics plan "
                 "covering transportation, transfers, and travel efficiency",
            backstory="A master planner who ensures smooth travel connections and optimal routing",
            verbose=True,
            llm=self.llm
        )
    
    def _create_destination_researcher(self):
        """Create destination research agent"""
        return Agent(
            role="Destination Intelligence Specialist",
            goal=f"Comprehensive research on {self.trip_details.to_destination} "
                 "considering budget, interests, and travel style",
            backstory="A global travel researcher with deep insights into destination specifics",
            verbose=True,
            llm=self.llm
        )
    
    def _create_itinerary_planner(self):
        """Create itinerary planning agent"""
        return Agent(
            role="Strategic Itinerary Architect",
            goal="Design a meticulously planned multi-destination travel itinerary "
                 "balancing cost, experiences, and traveler preferences",
            backstory="A master trip planner who crafts seamless multi-destination experiences",
            verbose=True,
            llm=self.llm
        )
    
    def _create_experience_curator(self):
        """Create experience personalization agent"""
        return Agent(
            role="Experience Personalization Maestro",
            goal="Refine and customize travel experiences to match exact traveler needs",
            backstory="A travel concierge who understands nuanced traveler desires across different destinations",
            verbose=True,
            llm=self.llm
        )
    
    def generate_comprehensive_trip_plan(self) -> Dict:
        """
        Generate a comprehensive multi-destination trip plan
        
        :return: Detailed trip plan with routing and itinerary
        """
        # Route Analysis Task
        route_analysis_task = Task(
            description=f"Comprehensive route analysis from {self.trip_details.from_destination} "
                        f"to {self.trip_details.to_destination}. Analyze: "
                        "- Optimal travel routes "
                        "- Transportation options "
                        "- Potential layovers or connections "
                        "- Travel time and efficiency",
            agent=self.route_analyzer,
            expected_output="Detailed route analysis and recommended travel paths"
        )
        
        # Travel Logistics Task
        logistics_task = Task(
            description=f"Develop comprehensive travel logistics for journey from "
                        f"{self.trip_details.from_destination} to {self.trip_details.to_destination}. "
                        "Considerations: "
                        f"- Budget range: ${self.trip_details.budget_min} - ${self.trip_details.budget_max} "
                        f"- Travel dates: {self.trip_details.travel_dates} "
                        "- Transportation recommendations "
                        "- Transfer strategies "
                        "- Visa and entry requirements",
            agent=self.travel_logistics_agent,
            expected_output="Comprehensive travel logistics and transportation plan"
        )
        
        # Destination Research Task
        research_task = Task(
            description=f"Deep research on {self.trip_details.to_destination}. Analyze: "
                        f"- Interests: {', '.join(self.trip_details.interests)} "
                        f"- Travel Style: {self.trip_details.travel_style} "
                        "- Local attractions "
                        "- Cultural insights "
                        "- Hidden gems matching traveler profile",
            agent=self.destination_researcher,
            expected_output="Comprehensive destination insights and recommended experiences"
        )
        
        # Itinerary Planning Task
        itinerary_task = Task(
            description=f"Create detailed itinerary for {self.trip_details.to_destination}. "
                        "Key requirements: "
                        f"- Origin: {self.trip_details.from_destination} "
                        f"- Budget: ${self.trip_details.budget_min} - ${self.trip_details.budget_max} "
                        f"- Travel dates: {self.trip_details.travel_dates} "
                        f"- Group size: {self.trip_details.group_size} "
                        f"- Interests: {', '.join(self.trip_details.interests)} "
                        "Deliverables: "
                        "- Day-by-day activity plan "
                        "- Cost breakdown "
                        "- Transportation details "
                        "- Local experiences",
            agent=self.itinerary_planner,
            expected_output="Comprehensive, budget-aligned multi-destination travel itinerary"
        )
        
        # Personalization Task
        personalization_task = Task(
            description="Final itinerary refinement. "
                        "Customize experiences, "
                        "ensure budget optimization, "
                        "add flexibility. "
                        f"Consider: {self.trip_details.dietary_restrictions}",
            agent=self.experience_curator,
            expected_output="Personalized, flexible, and optimized travel plan. Print only once",
            output_file="trip_plan.md"
        )
        
        # Create and execute crew
        crew = Crew(
            agents=[
                self.route_analyzer,
                self.travel_logistics_agent,
                self.destination_researcher,
                self.itinerary_planner,
                self.experience_curator
            ],
            tasks=[
                route_analysis_task,
                logistics_task,
                research_task,
                itinerary_task,
                personalization_task
            ],
            verbose=True
        )
        
        # Generate comprehensive trip plan
        result = crew.kickoff()
        
        return {
            "origin": self.trip_details.from_destination,
            "destination": self.trip_details.to_destination,
            "budget_range": {
                "min": self.trip_details.budget_min,
                "max": self.trip_details.budget_max
            },
            "travel_dates": self.trip_details.travel_dates,
            "comprehensive_trip_plan": str(result),
            "trip_details": asdict(self.trip_details)
        }

def main():
    # Comprehensive trip details example
    trip_details = TripDetails(
        from_destination="Hyderabad, India",  # Origin
        to_destination="Pondicherry, India",     # Destination
        budget_min=5000,                   # Minimum budget
        budget_max=7000,                   # Maximum budget
        interests=["Beach", "Church", "Historical", "Vegetarian Food"],
        travel_style="Adventure",
        group_size=11,
        travel_dates={
            "start_date": "2024-12-22",
            "end_date": "2024-12-25"
        },
        dietary_restrictions=["vegetarian"],
        accessibility_needs=["minimal walking"]
    )
    
    # Initialize and generate trip plan
    trip_planner = MultiDestinationTripPlanner(trip_details=trip_details)
    comprehensive_trip_plan = trip_planner.generate_comprehensive_trip_plan()
    
    # Save the result to a JSON file
    import json
    with open('comprehensive_trip_plan.json', 'w') as f:
        json.dump(comprehensive_trip_plan, f, indent=2)

if __name__ == "__main__":
    main()