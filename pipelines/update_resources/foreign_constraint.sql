/* When the table datasource is updated, make sure the foregin key constraints remain*/

ALTER TABLE dataset_has_datasource
DROP CONSTRAINT IF EXISTS "dataset_has_datasource_datasource_name_fkey";

ALTER TABLE dataset_has_datasource
ADD CONSTRAINT "dataset_has_datasource_datasource_name_fkey"
FOREIGN KEY (datasource_name)
REFERENCES datasource(datasource_name);

ALTER TABLE molrepo_has_datasource
DROP CONSTRAINT IF EXISTS "molrepo_has_datasource_datasource_name_fkey";

ALTER TABLE molrepo_has_datasource
ADD CONSTRAINT "molrepo_has_datasource_datasource_name_fkey"
FOREIGN KEY (datasource_name)
REFERENCES datasource(datasource_name);


