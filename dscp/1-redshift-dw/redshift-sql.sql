-- Part1. This is for copy raw dataset of identity to redshift table for further DW's use

create external schema dscp_identity_schema from data catalog 
database 'db-dscp-raw' 
iam_role 'arn:aws-cn:iam::542319707026:role/myredshiftrole'
create external database if not exists;

CREATE TABLE t_identity (LIKE dscp_identity_schema.raw_identity);

copy t_identity
from 's3://dscp/datasets/raw-identity/raw_identity.csv'
IGNOREHEADER 1
FILLRECORD
delimiter ','
iam_role 'arn:aws-cn:iam::542319707026:role/myredshiftrole';

-- Part2. This is for copy dataset processed by DW to a new Redshift table

CREATE TABLE t_processed_identity (LIKE dscp_identity_schema.dw_identitydw_output);

copy t_processed_identity
from 's3://dscp/for-redshift/dw-output/export-flow-11-05-58-56-8fb8580b/output/data-wrangler-flow-processing-11-05-58-56-8fb8580b/6e76dd05-fcfb-43ce-bc50-d8d55b9d1280/default/part-00000-967d8380-84f5-4f26-a6c9-523f7622ed5d-c000.csv'
IGNOREHEADER 1
FILLRECORD
delimiter ','
iam_role 'arn:aws-cn:iam::542319707026:role/myredshiftrole';

-- Part3. Drop Partitions if they exists

alter table t_processed_identity drop column partition_0;
alter table t_processed_identity drop column partition_1;
alter table t_processed_identity drop column partition_2;
alter table t_processed_identity drop column partition_3;
alter table t_processed_identity drop column partition_4;

