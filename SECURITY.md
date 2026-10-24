# Security policy

Do not cache secrets, credentials, payment data, medical information, or
personalised answers unless your deployment has an explicit retention,
encryption, access-control, and deletion policy.

Please report vulnerabilities privately through GitHub Security Advisories for
this repository. Do not include exploit details in public issues.

`CacheContext` is a partitioning helper, not an authorisation system. Enforce
tenant access control in the application and database layers.
