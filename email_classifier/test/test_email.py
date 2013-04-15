import os
import unittest
from ..email import Email, parse_email

body_text = """Jeff,

We will definitely need a lot more information related to the items below.  
Please let me know when we can meet and discuss these.  Some of the items we 
will need to know include:

We discussed discrepancies your team noted between the spreadsheet positions 
that are being captured and what's reported on the Gas BM report.  This is 
causing you to manually adjust the Gas BM report.  Is this for all 
spreadsheet positions?  Some of them?  Is the variance significant?  Is it 
off everyday?  
What do you mean by " All Transport Models".   Does this mean positions?  Are 
transport positions on the Gas BM report?  Are they manual adjustments?  Are 
they in GRMS?  Can we capture these with spreadsheets?
What is your current methodology in preparing the Gas BM report for 
roll-offs?  Is it currently consistent across all regions or is it handled 
differently by different traders?  Please explain this in further detail. The 
Basis roll-off was done in the GRMS system in the past.  Is this still the 
case?  If not, we need to capture these roll-off adjustments using 
spreadsheets?
Can we make the Fixed Price Canadian Deals adjustments using spreadsheets 
that can be captured in GRMS?
Also, what is the status of capturing IM-Canada positions in the system?

We will need this type of information urgently.  We are meeting with Jeff 
Shankman on Tuesday to update him on the progress we are making.  

Please let me know if there is anything we can do to help.

Thanks,
Mike


   
	Enron North America Corp.
	
	From:  Jeffrey C Gossett                           06/07/2000 05:26 PM
	

To: Michael E Moscoso/HOU/ECT@ECT
cc: Brent A Price/HOU/ECT@ECT, Steve Jackson/HOU/ECT@ECT, Susan 
Harrison/HOU/ECT@ECT, Sunil Dalal/Corp/Enron@ENRON, Vladimir 
Gorny/HOU/ECT@ECT 
Subject: Spreadsheets and other Data

Spreadsheets :
(in Quark)
EES_Dublin
NG_Opt
Pipe_Opt_date
Storage_opt_date

All Transport models:
East 
West
Central

Manual Adjustments:
Spreadsheets
Fixed Price Canadian Deals
Roll Off

Let me know if you need anything further.

Thanks

JG





"""

class TestEmail(unittest.TestCase):
	def setUp(self):
		self.email = Email(os.path.join(os.getcwd(), 'email_classifier/data/beck-s/moscoso__mike/4.'))

	def test_parse_email(self):
		fields = parse_email(os.path.join(os.getcwd(), 'email_classifier/data/beck-s/moscoso__mike/4.'))
		self.assertEqual(fields[0], 'Thu, 8 Jun 2000 11:24:00 -0700 (PDT)')
		self.assertEqual(fields[1], 'Michael E Moscoso')
		self.assertEqual(fields[2], 'Jeffrey C Gossett')
		self.assertEqual(fields[3], 'Brent A Price, Steve Jackson, Sunil Dalal, Vladimir Gorny')
		self.assertEqual(fields[4], '')
		self.assertEqual(fields[5], 'Re: Spreadsheets and other Data')
		self.assertEqual(fields[6], body_text)

	def test__extract_name(self):
		self.assertEqual(self.email._extract_name(self.email.header_to.split(',')[0]), 'Jeffrey C Gossett')

	def test__extract_names(self):
		names = self.email._extract_names(self.email.header_cc)
		self.assertEqual(names, ['Brent A Price', 'Steve Jackson', 'Sunil Dalal', 'Vladimir Gorny'])

	def test_get_from_names(self):
		pass

	def test_get_to_names(self):
		pass

	def test_get_cc_names(self):
		pass

	def test_get_bcc_names(self):
		pass

	def test_get_hour(self):
		self.assertEqual(self.email.get_hour(), 11)

	def test_get_month(self):
		self.assertEqual(self.email.get_month(), 'Jun')

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestEmail))
    return suite

if __name__ == '__main__':
    unittest.main()
