# Versioning Policy

The `turnkey` package applies semantic versioning for its 3-digit version number. The version number is stored in `src/version.py`.

The 3 digits correspond to MAJOR.MINOR.PATCH, which can be interpreted as follows:
* MAJOR: changes indicate breaking API changes that may require the user to change their own code
* MINOR: changes indicate that builds against a previous minor version may not be compatible, and the user may need to rebuild those models
* PATCH: no user action required when the patch number changes
