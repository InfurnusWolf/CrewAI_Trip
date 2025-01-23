import os
import requests
import json
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Set environment variable for API keys
openai_api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class TripDetails:
    """
    Comprehensive trip details with multi-destination support
    """
    from_destination: str
    to_destination: str
    budget_min: int
    budget_max: int
    interests: List[str]
    travel_style: str
    group_size: int
    travel_dates: Dict[str, str]
    dietary_restrictions: Optional[List[str]] = None
    accessibility_needs: Optional[List[str]] = None
    
    def validate(self):
        """
        Validate trip details
        """
        if self.budget_min <= 0 or self.budget_max <= 0:
            raise ValueError("Budget must be positive")
        
        if self.budget_min > self.budget_max:
            raise ValueError("Minimum budget cannot exceed maximum budget")
        
        if self.group_size <= 0:
            raise ValueError("Group size must be positive")
        
        if self.from_destination == self.to_destination:
            raise ValueError("Origin and destination cannot be the same")

# API Integration Functions
def get_flight_options(from_destination, to_destination, travel_dates, group_size):
    """Fetch flight options using a flights API"""
    url = "https://api.amadeus.com/v1/shopping/flight-offers"
    params = {
        "originLocationCode": from_destination,
        "destinationLocationCode": to_destination,
        "departureDate": travel_dates["start_date"],
        "returnDate": travel_dates["end_date"],
        "adults": group_size,
        # "currency": "USD",
        # "locale": "en-US",
    }
    headers = {"apikey": "fceb897cb1f5e1b951b463c01f1d7937"}
    response = requests.get(url, params=params, headers=headers)
    return response.json()

def get_hotel_options(destination, travel_dates, budget_range, group_size):
    """Fetch hotel options using a hotels API"""
    url = "https://hotels-com-free.p.rapidapi.com/v1/search"
    params = {
        "query": destination,
        "checkin_date": travel_dates["start_date"],
        "checkout_date": travel_dates["end_date"],
        "adults": group_size,
        "price_min": budget_range["min"],
        "price_max": budget_range["max"],
    }
    headers = {
        "X-RapidAPI-Key": "3ff7877a8fmsha1bc83c54feab8ap13616djsn94ac3deec2bc",
        "X-RapidAPI-Host": "tripadvisor16.p.rapidapi.com",
    }
    response = requests.get(url, params=params, headers=headers)
    return response.json()

def get_local_activities(destination, interests):
    """Fetch local activities using a local activities API"""
    url = f"https://api.geoapify.com/v2/places"
    params = {
        "query": destination,
        "categories": ",".join(interests),
    }
    headers = {"apikey": "1b33c270eae94618aafaed17fbba29ef"}
    response = requests.get(url, params=params, headers=headers)
    return response.json()

def get_weather_forecast(destination, travel_dates):
    """Fetch weather forecast using a weather API"""
    url = f"https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": destination,
        "units": "metric",
        "appid": "fb8ee50b6fbc6fc967e6211f6f5406cb",
    }
    response = requests.get(url, params=params)
    return response.json()

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
    
    def generate_comprehensive_trip_plan(self):
        # Fetch data from APIs
        flight_options = get_flight_options(
            self.trip_details.from_destination,
            self.trip_details.to_destination,
            self.trip_details.travel_dates,
            self.trip_details.group_size
        )
        hotel_options = get_hotel_options(
            self.trip_details.to_destination,
            self.trip_details.travel_dates,
            {
                "min": self.trip_details.budget_min,
                "max": self.trip_details.budget_max
            },
            self.trip_details.group_size
        )
        local_activities = get_local_activities(
            self.trip_details.to_destination,
            self.trip_details.interests
        )
        weather_forecast = get_weather_forecast(
            self.trip_details.to_destination,
            self.trip_details.travel_dates
        )
        
        # Combine results into a trip plan
        return {
            "origin": self.trip_details.from_destination,
            "destination": self.trip_details.to_destination,
            "budget_range": {
                "min": self.trip_details.budget_min,
                "max": self.trip_details.budget_max
            },
            "travel_dates": self.trip_details.travel_dates,
            "flights": flight_options,
            "hotels": hotel_options,
            "activities": local_activities,
            "weather_forecast": weather_forecast,
            "trip_details": asdict(self.trip_details)
        }

def main():
    # Example trip details
    trip_details = TripDetails(
        from_destination="New York, USA",
        to_destination="Tokyo, Japan",
        budget_min=3000,
        budget_max=6000,
        interests=["technology", "culture", "food"],
        travel_style="immersive",
        group_size=2,
        travel_dates={
            "start_date": "2024-09-15",
            "end_date": "2024-09-25"
        },
        dietary_restrictions=["vegetarian"],
        accessibility_needs=["minimal walking"]
    )
    
    # Generate trip plan
    trip_planner = MultiDestinationTripPlanner(trip_details=trip_details)
    comprehensive_trip_plan = trip_planner.generate_comprehensive_trip_plan()
    
    # Save to JSON file
    with open('comprehensive_trip_plan.json', 'w') as f:
        json.dump(comprehensive_trip_plan, f, indent=2)

if __name__ == "__main__":
    main()
