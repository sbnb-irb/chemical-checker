DROP TABLE IF EXISTS datasourcebckp;

CREATE TABLE datasourcebckp AS TABLE datasource;

ALTER TABLE dataset_has_datasource
DROP CONSTRAINT IF EXISTS "dataset_has_datasource_datasource_name_fkey";

ALTER TABLE molrepo_has_datasource
DROP CONSTRAINT IF EXISTS "molrepo_has_datasource_datasource_name_fkey";



