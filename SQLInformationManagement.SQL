CREATE TABLE Client (
    Client_ID INT PRIMARY KEY,
    Name VARCHAR(100),
    Num_Employees INT,
    Primary_Contact_Name VARCHAR(100),
    Email VARCHAR(100),
    Date_signed DATE,
    Plan VARCHAR(50)
);

CREATE TABLE Workspace (
    Workspace_ID INT PRIMARY KEY,
    Client_ID INT,
    Name VARCHAR(100),
    is_Active BOOLEAN,
    sel_conf VARCHAR(100),
    Updated_Date DATE,
    Create_Date DATE,
    App_ID INT,
    Channel_Type VARCHAR(50),
    FOREIGN KEY (Client_ID) REFERENCES Client(Client_ID)
);

CREATE TABLE Companies (
    Company_ID INT PRIMARY KEY,
    sel_conf VARCHAR(100),
    Deployment_Status VARCHAR(50),
    Last_seen DATE,
    App_ID INT
);

CREATE TABLE Contacts (
    Contact_ID INT PRIMARY KEY,
    Company_ID INT,
    Name VARCHAR(100),
    Last_seen DATE,
    Phone VARCHAR(20),
    Email VARCHAR(100),
    FOREIGN KEY (Company_ID) REFERENCES Companies(Company_ID)
);

CREATE TABLE Services (
    Service_ID INT PRIMARY KEY,
    Workspace_ID INT,
    FOREIGN KEY (Workspace_ID) REFERENCES Workspace(Workspace_ID)
);

CREATE TABLE Members (
    Member_ID INT PRIMARY KEY,
    Workspace_ID INT,
    Name VARCHAR(100),
    Email VARCHAR(100),
    Role VARCHAR(50),
    State VARCHAR(50),
    FOREIGN KEY (Workspace_ID) REFERENCES Workspace(Workspace_ID)
);

CREATE TABLE Service_Team (
    Team_ID INT PRIMARY KEY,
    Name VARCHAR(100),
    Manager INT,
    FOREIGN KEY (Manager) REFERENCES Members(Member_ID)
);




CREATE TABLE Member_Management (
    Manager_ID INT,
    Managee_ID INT,
    PRIMARY KEY (Manager_ID, Managee_ID),
    FOREIGN KEY (Manager_ID) REFERENCES Members(Member_ID),
    FOREIGN KEY (Managee_ID) REFERENCES Members(Member_ID)
);

CREATE TABLE Member_Service_Team (
    Member_ID INT,
    Team_ID INT,
    PRIMARY KEY (Member_ID, Team_ID),
    FOREIGN KEY (Member_ID) REFERENCES Members(Member_ID),
    FOREIGN KEY (Team_ID) REFERENCES Service_Team(Team_ID)
);







-- Populate Client table
INSERT INTO Client (Client_ID, Name, Num_Employees, Primary_Contact_Name, Email, Date_signed, Plan)
VALUES
(21337801, 'Alpha Corporation', 26, 'Bill Gates', 'billyboy@gmail.gov', '2022-01-15', 'Premium'),
(2, 'Beta Inc.', 80, 'Jane Smith', 'janesmith@beta.com', '2024-02-20', 'Standard'),
(3, 'Gamma LLC', 50, 'Alice Johnson', 'alice@gamma.com', '2024-03-05', 'Basic'),
(4, 'Delta Ltd', 200, 'Bob Brown', 'bob@delta.com', '2024-04-10', 'Enterprise');

-- Populate Workspace table
INSERT INTO Workspace (Workspace_ID, Client_ID, Name, is_Active, sel_conf, Updated_Date, Create_Date, App_ID, Channel_Type)
VALUES
(1, 21337801, 'Alpha Workspace', TRUE, 'Teams', '2024-01-16', '2024-01-15', 101, 'Public'),
(2, 2, 'Beta Workspace', TRUE, 'Slack', '2024-02-21', '2024-02-20', 102, 'Private'),
(3, 3, 'Gamma Workspace', FALSE, 'Slack', '2024-03-06', '2024-03-05', 103, 'Public'),
(4, 4, 'Delta Workspace', TRUE, 'Teams', '2024-04-11', '2024-04-10', 104, 'Private');

-- Populate Companies table
INSERT INTO Companies (Company_ID, sel_conf, Deployment_Status, Last_seen, App_ID)
VALUES
(1, 'Teams', 'Deployed', '2024-01-20', 101),
(2, 'Teams', 'Pending', '2024-02-25', 102),
(3, 'Slack', 'Failed', '2024-03-10', 103),
(4, 'Slack', 'Deployed', '2024-04-15', 104);

-- Populate Contacts table
INSERT INTO Contacts (Contact_ID, Company_ID, Name, Last_seen, Phone, Email)
VALUES
(1, 1, 'Charlie Alpha', '2024-01-22', '555-0101', 'charlie@alpha.com'),
(2, 2, 'Diana Beta', '2024-02-28', '555-0102', 'diana@beta.com'),
(3, 3, 'Edward Gamma', '2024-03-12', '555-0103', 'edward@gamma.com'),
(4, 4, 'Fiona Delta', '2024-04-18', '555-0104', 'fiona@delta.com');

-- Populate Services table
INSERT INTO Services (Service_ID, Workspace_ID)
VALUES
(1, 1),
(2, 2),
(3, 3),
(4, 4);

-- Populate Members table
INSERT INTO Members (Member_ID, Workspace_ID, Name, Email, Role, State)
VALUES
(1, 1, 'Gina Alpha', 'gina@alpha.com', 'Admin', 'Active'),
(2, 2, 'Henry Beta', 'henry@beta.com', 'User', 'Active'),
(3, 3, 'Isabel Gamma', 'isabel@gamma.com', 'User', 'Inactive'),
(4, 4, 'Jack Delta', 'jack@delta.com', 'Admin', 'Active');

-- Populate Service_Team table
INSERT INTO Service_Team (Team_ID, Name, Manager)
VALUES
(1, 'Alpha Team', 1),
(2, 'Beta Team', 2),
(3, 'Gamma Team', 3),
(4, 'Delta Team', 4);

-- Populate Member_Management table (for "manages" relationship)
INSERT INTO Member_Management (Manager_ID, Managee_ID)
VALUES
(1, 2),
(1, 3),
(2, 4),
(3, 1);

-- Populate Member_Service_Team table (for "works on" relationship)
INSERT INTO Member_Service_Team (Member_ID, Team_ID)
VALUES
(1, 1),
(2, 2),
(3, 3),
(4, 4);



 -- (a) used to find all entities of a particular entity type
SELECT * FROM Client;

 -- (b) used to return all entities of a particular entity type
SELECT * FROM Client WHERE Plan = 'Premium';
 -- or
SELECT * FROM Workspace WHERE is_Active = TRUE;

 -- (c) used to return entities of a 1:N relationship
SELECT * FROM Contacts WHERE Company_ID = 2;
