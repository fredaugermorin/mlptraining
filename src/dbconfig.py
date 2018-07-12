import pyodbc as pdb 

# you can setup your database access here
connections_ =  {'sqlserver':pdb.connect(
                            r'DRIVER={SQL Server};'
                            r'SERVER=DESKTOP-RR00UM3;'
                            r'DATABASE=MastersThesis;'
                            r'Trusted_Connection=yes;'
                            )
                }
