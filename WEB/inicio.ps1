.\Activate.ps1
Set-Location ..\..
$env:FLASK_APP = "index.py"
$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = 1
flask run