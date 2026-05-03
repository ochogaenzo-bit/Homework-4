"""
Customer Service AI Agent
Handles customer data parsing, validation, statistics, and reporting
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TicketStatus(Enum):
    """Enumeration for ticket status"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Priority(Enum):
    """Enumeration for ticket priority"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class CustomerTicket:
    """Data class for customer support tickets"""
    ticket_id: str
    customer_name: str
    email: str
    issue_type: str
    description: str
    priority: str
    status: str
    created_date: str
    resolved_date: Optional[str] = None


class CustomerServiceAgent:
    """Main class for customer service AI agent"""
    
    def __init__(self):
        self.tickets: List[CustomerTicket] = []
        self.validation_errors: List[str] = []
        self.statistics: Dict = {}
    
    # ==================== PARSING A FILE ====================
    
    def parse_file(self, filename: str, file_format: str = 'json') -> bool:
        """
        Parse customer service data from a file
        Supports JSON and CSV formats
        """
        try:
            if file_format.lower() == 'json':
                return self._parse_json(filename)
            elif file_format.lower() == 'csv':
                return self._parse_csv(filename)
            else:
                print(f"Unsupported file format: {file_format}")
                return False
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
            return False
        except Exception as e:
            print(f"Error parsing file: {str(e)}")
            return False
    
    def _parse_json(self, filename: str) -> bool:
        """Parse JSON format file"""
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            if isinstance(data, list):
                tickets_data = data
            elif isinstance(data, dict) and 'tickets' in data:
                tickets_data = data['tickets']
            else:
                print("Invalid JSON structure")
                return False
            
            for ticket_data in tickets_data:
                ticket = self._extract_fields(ticket_data)
                if ticket:
                    self.tickets.append(ticket)
            
            print(f"Successfully parsed {len(self.tickets)} tickets from {filename}")
            return True
    
    def _parse_csv(self, filename: str) -> bool:
        """Parse CSV format file"""
        import csv
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                ticket = self._extract_fields(row)
                if ticket:
                    self.tickets.append(ticket)
            
            print(f"Successfully parsed {len(self.tickets)} tickets from {filename}")
            return True
    
    # ==================== EXTRACTING FIELDS ====================
    
    def _extract_fields(self, data: Dict) -> Optional[CustomerTicket]:
        """
        Extract and map fields from raw data to CustomerTicket object
        """
        try:
            ticket = CustomerTicket(
                ticket_id=str(data.get('ticket_id', data.get('id', ''))),
                customer_name=data.get('customer_name', data.get('name', '')),
                email=data.get('email', data.get('customer_email', '')),
                issue_type=data.get('issue_type', data.get('type', '')),
                description=data.get('description', data.get('issue', '')),
                priority=data.get('priority', 'medium').lower(),
                status=data.get('status', 'open').lower(),
                created_date=data.get('created_date', data.get('date', '')),
                resolved_date=data.get('resolved_date', data.get('resolution_date'))
            )
            return ticket
        except Exception as e:
            print(f"Error extracting fields: {str(e)}")
            return None
    
    # ==================== VALIDATING STRUCTURE ====================
    
    def validate_structure(self) -> Tuple[bool, List[str]]:
        """
        Validate the structure and content of parsed tickets
        Returns tuple of (is_valid, list_of_errors)
        """
        self.validation_errors = []
        
        for i, ticket in enumerate(self.tickets):
            errors = []
            
            # Validate ticket ID
            if not ticket.ticket_id or ticket.ticket_id.strip() == '':
                errors.append(f"Ticket {i}: Missing ticket ID")
            
            # Validate customer name
            if not ticket.customer_name or len(ticket.customer_name.strip()) < 2:
                errors.append(f"Ticket {ticket.ticket_id}: Invalid customer name")
            
            # Validate email format
            if not self._validate_email(ticket.email):
                errors.append(f"Ticket {ticket.ticket_id}: Invalid email format")
            
            # Validate issue type
            if not ticket.issue_type or ticket.issue_type.strip() == '':
                errors.append(f"Ticket {ticket.ticket_id}: Missing issue type")
            
            # Validate description
            if not ticket.description or len(ticket.description.strip()) < 10:
                errors.append(f"Ticket {ticket.ticket_id}: Description too short or missing")
            
            # Validate priority
            valid_priorities = [p.value for p in Priority]
            if ticket.priority not in valid_priorities:
                errors.append(f"Ticket {ticket.ticket_id}: Invalid priority '{ticket.priority}'")
            
            # Validate status
            valid_statuses = [s.value for s in TicketStatus]
            if ticket.status not in valid_statuses:
                errors.append(f"Ticket {ticket.ticket_id}: Invalid status '{ticket.status}'")
            
            # Validate date format
            if not self._validate_date(ticket.created_date):
                errors.append(f"Ticket {ticket.ticket_id}: Invalid created date format")
            
            self.validation_errors.extend(errors)
        
        is_valid = len(self.validation_errors) == 0
        
        if is_valid:
            print("✓ All tickets validated successfully")
        else:
            print(f"✗ Validation failed with {len(self.validation_errors)} errors")
        
        return is_valid, self.validation_errors
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format using regex"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_date(self, date_str: str) -> bool:
        """Validate date format (supports multiple formats)"""
        date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%m/%d/%Y']
        for fmt in date_formats:
            try:
                datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
        return False
    
    # ==================== COMPUTING STATISTICS ====================
    
    def compute_statistics(self) -> Dict:
        """
        Compute various statistics from the ticket data
        """
        if not self.tickets:
            print("No tickets to compute statistics")
            return {}
        
        stats = {
            'total_tickets': len(self.tickets),
            'by_status': self._count_by_field('status'),
            'by_priority': self._count_by_field('priority'),
            'by_issue_type': self._count_by_field('issue_type'),
            'resolution_rate': self._calculate_resolution_rate(),
            'average_response_time': self._calculate_avg_response_time(),
            'top_customers': self._get_top_customers(5),
            'urgent_unresolved': self._count_urgent_unresolved()
        }
        
        self.statistics = stats
        print("✓ Statistics computed successfully")
        return stats
    
    def _count_by_field(self, field: str) -> Dict[str, int]:
        """Count tickets by a specific field"""
        counts = {}
        for ticket in self.tickets:
            value = getattr(ticket, field, 'Unknown')
            counts[value] = counts.get(value, 0) + 1
        return counts
    
    def _calculate_resolution_rate(self) -> float:
        """Calculate percentage of resolved/closed tickets"""
        resolved_count = sum(1 for t in self.tickets 
                           if t.status in ['resolved', 'closed'])
        return round((resolved_count / len(self.tickets)) * 100, 2)
    
    def _calculate_avg_response_time(self) -> str:
        """Calculate average time to resolve tickets"""
        resolved_tickets = [t for t in self.tickets if t.resolved_date]
        
        if not resolved_tickets:
            return "N/A"
        
        total_days = 0
        valid_count = 0
        
        for ticket in resolved_tickets:
            try:
                created = self._parse_date(ticket.created_date)
                resolved = self._parse_date(ticket.resolved_date)
                if created and resolved:
                    delta = (resolved - created).days
                    total_days += delta
                    valid_count += 1
            except:
                continue
        
        if valid_count == 0:
            return "N/A"
        
        avg_days = total_days / valid_count
        return f"{avg_days:.1f} days"
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%m/%d/%Y']
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    
    def _get_top_customers(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get top N customers by ticket count"""
        customer_counts = {}
        for ticket in self.tickets:
            customer_counts[ticket.customer_name] = \
                customer_counts.get(ticket.customer_name, 0) + 1
        
        sorted_customers = sorted(customer_counts.items(), 
                                 key=lambda x: x[1], reverse=True)
        return sorted_customers[:n]
    
    def _count_urgent_unresolved(self) -> int:
        """Count urgent tickets that are not resolved"""
        return sum(1 for t in self.tickets 
                  if t.priority == 'urgent' and 
                  t.status not in ['resolved', 'closed'])
    
    # ==================== TRANSFORMING DATA ====================
    
    def transform_data(self) -> List[Dict]:
        """
        Transform ticket data for different purposes
        (e.g., normalize, enrich, or format for external systems)
        """
        transformed = []
        
        for ticket in self.tickets:
            transformed_ticket = {
                'id': ticket.ticket_id,
                'customer': {
                    'name': ticket.customer_name.title(),
                    'email': ticket.email.lower(),
                    'contact_method': 'email'
                },
                'issue': {
                    'type': ticket.issue_type.replace('_', ' ').title(),
                    'description': ticket.description,
                    'priority_level': self._map_priority_to_number(ticket.priority),
                    'priority_label': ticket.priority.upper()
                },
                'status': {
                    'current': ticket.status.replace('_', ' ').title(),
                    'is_open': ticket.status in ['open', 'in_progress'],
                    'is_resolved': ticket.status in ['resolved', 'closed']
                },
                'dates': {
                    'created': ticket.created_date,
                    'resolved': ticket.resolved_date,
                    'age_days': self._calculate_ticket_age(ticket.created_date)
                },
                'metadata': {
                    'requires_attention': self._requires_attention(ticket),
                    'sentiment': self._analyze_sentiment(ticket.description)
                }
            }
            transformed.append(transformed_ticket)
        
        print(f"✓ Transformed {len(transformed)} tickets")
        return transformed
    
    def _map_priority_to_number(self, priority: str) -> int:
        """Map priority string to numeric value"""
        priority_map = {'low': 1, 'medium': 2, 'high': 3, 'urgent': 4}
        return priority_map.get(priority, 2)
    
    def _calculate_ticket_age(self, created_date: str) -> int:
        """Calculate how many days old the ticket is"""
        try:
            created = self._parse_date(created_date)
            if created:
                return (datetime.now() - created).days
        except:
            pass
        return 0
    
    def _requires_attention(self, ticket: CustomerTicket) -> bool:
        """Determine if ticket requires immediate attention"""
        if ticket.priority == 'urgent' and ticket.status != 'resolved':
            return True
        
        age = self._calculate_ticket_age(ticket.created_date)
        if age > 7 and ticket.status == 'open':
            return True
        
        return False
    
    def _analyze_sentiment(self, description: str) -> str:
        """Basic sentiment analysis of ticket description"""
        negative_words = ['angry', 'frustrated', 'terrible', 'awful', 
                         'horrible', 'worst', 'hate', 'broken', 'useless']
        positive_words = ['thank', 'please', 'appreciate', 'great', 'good']
        
        description_lower = description.lower()
        
        negative_count = sum(1 for word in negative_words 
                            if word in description_lower)
        positive_count = sum(1 for word in positive_words 
                            if word in description_lower)
        
        if negative_count > positive_count:
            return 'negative'
        elif positive_count > negative_count:
            return 'positive'
        else:
            return 'neutral'
    
    # ==================== FORMATTING A REPORT ====================
    
    def format_report(self, output_file: str = 'customer_service_report.txt') -> str:
        """
        Generate and format a comprehensive report
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 70)
        report_lines.append("CUSTOMER SERVICE AI AGENT - ANALYTICS REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Tickets Analyzed: {len(self.tickets)}")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Statistics Section
        if self.statistics:
            report_lines.append("📊 STATISTICS SUMMARY")
            report_lines.append("-" * 70)
            report_lines.append(f"Total Tickets: {self.statistics['total_tickets']}")
            report_lines.append(f"Resolution Rate: {self.statistics['resolution_rate']}%")
            report_lines.append(f"Average Response Time: {self.statistics['average_response_time']}")
            report_lines.append(f"Urgent Unresolved: {self.statistics['urgent_unresolved']}")
            report_lines.append("")
            
            # Status Breakdown
            report_lines.append("📋 STATUS BREAKDOWN")
            report_lines.append("-" * 70)
            for status, count in self.statistics['by_status'].items():
                percentage = (count / self.statistics['total_tickets']) * 100
                report_lines.append(f"  {status.title():20s}: {count:3d} ({percentage:5.1f}%)")
            report_lines.append("")
            
            # Priority Breakdown
            report_lines.append("⚠️  PRIORITY BREAKDOWN")
            report_lines.append("-" * 70)
            for priority, count in sorted(self.statistics['by_priority'].items(),
                                         key=lambda x: self._map_priority_to_number(x[0]),
                                         reverse=True):
                percentage = (count / self.statistics['total_tickets']) * 100
                report_lines.append(f"  {priority.upper():20s}: {count:3d} ({percentage:5.1f}%)")
            report_lines.append("")
            
            # Issue Type Breakdown
            report_lines.append("🔧 ISSUE TYPE BREAKDOWN")
            report_lines.append("-" * 70)
            sorted_issues = sorted(self.statistics['by_issue_type'].items(),
                                  key=lambda x: x[1], reverse=True)
            for issue_type, count in sorted_issues[:10]:  # Top 10
                percentage = (count / self.statistics['total_tickets']) * 100
                report_lines.append(f"  {issue_type.title():30s}: {count:3d} ({percentage:5.1f}%)")
            report_lines.append("")
            
            # Top Customers
            report_lines.append("👥 TOP CUSTOMERS (by ticket count)")
            report_lines.append("-" * 70)
            for i, (customer, count) in enumerate(self.statistics['top_customers'], 1):
                report_lines.append(f"  {i}. {customer:30s}: {count:3d} tickets")
            report_lines.append("")
        
        # Validation Errors (if any)
        if self.validation_errors:
            report_lines.append("⚠️  VALIDATION ERRORS")
            report_lines.append("-" * 70)
            for error in self.validation_errors[:20]:  # Show first 20
                report_lines.append(f"  • {error}")
            if len(self.validation_errors) > 20:
                report_lines.append(f"  ... and {len(self.validation_errors) - 20} more errors")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("💡 RECOMMENDATIONS")
        report_lines.append("-" * 70)
        report_lines.extend(self._generate_recommendations())
        report_lines.append("")
        
        report_lines.append("=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)
        
        report_text = "\n".join(report_lines)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ Report generated and saved to '{output_file}'")
        return report_text
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on data"""
        recommendations = []
        
        if not self.statistics:
            return ["• Run compute_statistics() first to generate recommendations"]
        
        # Check resolution rate
        if self.statistics['resolution_rate'] < 70:
            recommendations.append(
                f"• Resolution rate is {self.statistics['resolution_rate']}% - "
                "Consider increasing support staff or improving processes"
            )
        
        # Check urgent tickets
        if self.statistics['urgent_unresolved'] > 0:
            recommendations.append(
                f"• {self.statistics['urgent_unresolved']} urgent tickets need immediate attention"
            )
        
        # Check open tickets
        open_count = self.statistics['by_status'].get('open', 0)
        if open_count > len(self.tickets) * 0.3:
            recommendations.append(
                f"• {open_count} open tickets ({(open_count/len(self.tickets)*100):.1f}%) - "
                "Focus on reducing backlog"
            )
        
        # Check issue types
        if self.statistics['by_issue_type']:
            top_issue = max(self.statistics['by_issue_type'].items(), 
                          key=lambda x: x[1])
            if top_issue[1] > len(self.tickets) * 0.3:
                recommendations.append(
                    f"• '{top_issue[0]}' represents {top_issue[1]} tickets - "
                    "Consider creating FAQ or documentation"
                )
        
        if not recommendations:
            recommendations.append("• No critical issues detected - keep up the good work!")
        
        return recommendations
    
    # ==================== ADDITIONAL UTILITY METHODS ====================
    
    def search_tickets(self, keyword: str, field: str = 'all') -> List[CustomerTicket]:
        """Search tickets by keyword in specified field"""
        keyword_lower = keyword.lower()
        results = []
        
        for ticket in self.tickets:
            if field == 'all':
                search_text = f"{ticket.customer_name} {ticket.email} {ticket.description} {ticket.issue_type}".lower()
                if keyword_lower in search_text:
                    results.append(ticket)
            elif hasattr(ticket, field):
                field_value = str(getattr(ticket, field)).lower()
                if keyword_lower in field_value:
                    results.append(ticket)
        
        return results
    
    def export_tickets(self, filename: str, format: str = 'json'):
        """Export tickets to file"""
        if format == 'json':
            ticket_dicts = [vars(ticket) for ticket in self.tickets]
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(ticket_dicts, f, indent=2)
            print(f"✓ Exported {len(self.tickets)} tickets to {filename}")
        elif format == 'csv':
            import csv
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                if self.tickets:
                    fieldnames = vars(self.tickets[0]).keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for ticket in self.tickets:
                        writer.writerow(vars(ticket))
            print(f"✓ Exported {len(self.tickets)} tickets to {filename}")
    
    def print_summary(self):
        """Print a quick summary to console"""
        print("\n" + "=" * 50)
        print("CUSTOMER SERVICE AGENT - QUICK SUMMARY")
        print("=" * 50)
        print(f"Total Tickets: {len(self.tickets)}")
        
        if self.statistics:
            print(f"Resolution Rate: {self.statistics['resolution_rate']}%")
            print(f"Urgent Unresolved: {self.statistics['urgent_unresolved']}")
            
            print("\nStatus Distribution:")
            for status, count in self.statistics['by_status'].items():
                print(f"  {status.title()}: {count}")
        
        print("=" * 50 + "\n")


# ==================== MAIN FUNCTION / DEMO ====================

def main():
    """
    Main function demonstrating all features of the Customer Service AI Agent
    """
    print("🤖 Customer Service AI Agent - Starting...\n")
    
    # Create sample data file for demonstration
    sample_data = {
        "tickets": [
            {
                "ticket_id": "T001",
                "customer_name": "John Doe",
                "email": "john.doe@example.com",
                "issue_type": "billing_issue",
                "description": "I was charged twice for my last order. This is very frustrating and needs to be resolved immediately.",
                "priority": "urgent",
                "status": "open",
                "created_date": "2024-01-15"
            },
            {
                "ticket_id": "T002",
                "customer_name": "Jane Smith",
                "email": "jane.smith@example.com",
                "issue_type": "technical_support",
                "description": "The application keeps crashing when I try to upload files. Please help.",
                "priority": "high",
                "status": "in_progress",
                "created_date": "2024-01-16",
                "resolved_date": "2024-01-18"
            },
            {
                "ticket_id": "T003",
                "customer_name": "Bob Johnson",
                "email": "bob.j@example.com",
                "issue_type": "account_access",
                "description": "I forgot my password and cannot reset it. Thank you for your help.",
                "priority": "medium",
                "status": "resolved",
                "created_date": "2024-01-10",
                "resolved_date": "2024-01-11"
            },
            {
                "ticket_id": "T004",
                "customer_name": "Alice Williams",
                "email": "alice.w@example.com",
                "issue_type": "feature_request",
                "description": "It would be great if you could add dark mode to the application.",
                "priority": "low",
                "status": "open",
                "created_date": "2024-01-12"
            },
            {
                "ticket_id": "T005",
                "customer_name": "John Doe",
                "email": "john.doe@example.com",
                "issue_type": "shipping_delay",
                "description": "My order hasn't arrived yet and it's been a week. This is terrible service!",
                "priority": "high",
                "status": "in_progress",
                "created_date": "2024-01-14"
            },
            {
                "ticket_id": "T006",
                "customer_name": "Sarah Davis",
                "email": "sarah.davis@example.com",
                "issue_type": "product_defect",
                "description": "The product I received is broken. Need replacement as soon as possible.",
                "priority": "urgent",
                "status": "open",
                "created_date": "2024-01-17"
            }
        ]
    }
    
    # Save sample data to file
    with open('customer_tickets.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("✓ Sample data file created: customer_tickets.json\n")
    
    # Initialize the agent
    agent = CustomerServiceAgent()
    
    # 1. PARSING A FILE
    print("1️⃣  PARSING FILE...")
    print("-" * 50)
    agent.parse_file('customer_tickets.json', 'json')
    print()
    
    # 2. VALIDATING STRUCTURE
    print("2️⃣  VALIDATING STRUCTURE...")
    print("-" * 50)
    is_valid, errors = agent.validate_structure()
    if errors:
        print("Validation errors found:")
        for error in errors[:5]:  # Show first 5
            print(f"  - {error}")
    print()
    
    # 3. COMPUTING STATISTICS
    print("3️⃣  COMPUTING STATISTICS...")
    print("-" * 50)
    stats = agent.compute_statistics()
    print()
    
    # 4. TRANSFORMING DATA
    print("4️⃣  TRANSFORMING DATA...")
    print("-" * 50)
    transformed = agent.transform_data()
    print(f"Sample transformed ticket:")
    print(json.dumps(transformed[0], indent=2))
    print()
    
    # 5. FORMATTING REPORT
    print("5️⃣  FORMATTING REPORT...")
    print("-" * 50)
    report = agent.format_report('customer_service_report.txt')
    print()
    
    # Display summary
    agent.print_summary()
    
    # Additional demonstrations
    print("6️⃣  ADDITIONAL FEATURES...")
    print("-" * 50)
    
    # Search functionality
    urgent_tickets = agent.search_tickets('urgent', 'priority')
    print(f"Found {len(urgent_tickets)} urgent tickets")
    
    # Export functionality
    agent.export_tickets('exported_tickets.json', 'json')
    
    print("\n✅ All operations completed successfully!")
    print("\nGenerated files:")
    print("  - customer_tickets.json (sample data)")
    print("  - customer_service_report.txt (formatted report)")
    print("  - exported_tickets.json (exported data)")


if __name__ == "__main__":
    main()
