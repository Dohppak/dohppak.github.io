/* eslint-disable max-len, react/jsx-one-expression-per-line */
import React from 'react';
import { graphql } from 'gatsby';
import PropTypes from 'prop-types';
import Helmet from 'react-helmet';
import styled from '@emotion/styled';
import Layout from '../components/layout';
import siteShape from '../shapes/site';
import excelSetupPng from '../images/excel-setup-diagram.png';

const maLink = <a href="https://www.youtube.com/watch?v=JvUMV1N7eGM">Massachusetts</a>;
const ghLink = <a href="https://github.com/knpwrs">my GitHub</a>;
const patsLink = <a href="http://www.patriots.com/">New England Patriots</a>;
const cdpLink = <a href="http://cursordanceparty.com">Cursor Dance Party</a>;
const esdLink = <a href={excelSetupPng}>full setup diagram</a>;

const ResumeHeader = styled.header(({ theme }) => ({
  ...theme.centerPadding,
  display: 'flex',
  flexDirection: 'row',
  justifyContent: 'space-between',
  '> h5': {
    margin: '2em 0 1em 0',
  },
}));

const H2 = styled.h2(({ theme }) => ({
  ...theme.centerPadding,
  marginBottom: theme.spacing,
}));
const H3 = styled.h3(({ theme }) => ({
  fontFamily: 'Times',
  fontWeight: 'bold',
  fontStyle: 'italic',
  fontSize: '1.25em',
  ...theme.centerPadding,
  marginBottom: theme.spacing,
}));
const H4 = styled.h4(({ theme }) => ({
  ...theme.centerPadding,
  marginBottom: theme.spacing,
}));
const P = styled.p(({ theme }) => ({
  fontSize: '0.9em',
  ...theme.centerPadding,
}));

const Ul = styled.ul(({ theme }) => ({
  fontSize: '0.9em',
  ...theme.centerPadding,
  marginBottom: theme.spacing,
  marginLeft: `${theme.spacingPx * 4}px`,
}));

const About = ({ data: { site: { siteMetadata: site } } }) => (
  <Layout>
    <main>
      <Helmet>
        <title>
          About
          {' '}
          &middot;
          {' '}
          {site.title}
        </title>
      </Helmet>
      <H2>About</H2>
      <blockquote>
        <P>
          &quot;Far away across the oceans. An undiscovered paradise. Forget New
          York and California. There’s a better place – now close your eyes. Take
          my hand.  We are almost there. Our favorite place on Earth.&quot; -
          Ylvis
        </P>
      </blockquote>
      <P>
        I am a software engineer living and working in {maLink}. I work
        extensively in Universal JavaScript and HTML5 and have experience in many
        other technologies. Take a look at {ghLink} to see my personal projects.
      </P>
      <P>
        I also enjoy music, play drums and bass guitar, and am a big time fan of
        the {patsLink}. Feel free to take a look around and contact me with any
        questions.
      </P>
      <H3>Résumé</H3>
      <H4>Languages</H4>
      <Ul>
        <li>Proficient in: JavaScript (Universal Node / Browser, TypeScript, React), HTML5, CSS3 (SCSS)</li>
        <li>Familiar with: C# and .NET Framework, Java, Scala, Ruby, Swift, Rust, SQL</li>
      </Ul>
      <H4>Software</H4>
      <Ul>
        <li>Database: PostgreSQL, MySQL, SQL Server, MongoDB, Redis</li>
        <li>Server: nginx, Apache httpd</li>
        <li>Tools: Docker, Git, Jenkins, Travis CI</li>
        <li>Platforms: macOS, Linux / Unix, Microsoft Windows</li>
      </Ul>
      <H4>Experience</H4>
      <ResumeHeader>
        <h4>SHIFT Media &middot; Front End Lead &middot; Boston, MA</h4>
        <h5>September 2017 - Present</h5>
      </ResumeHeader>
      <Ul>
        <li>Worked with React, Redux, Electron, and modern JavaScript (TypeScript, Flow, JSX).</li>
        <li>Designed and implemented WebSocket communication layer with sagas.</li>
        <li>Designed and implemented generic upload queueing system with sagas.</li>
        <li>Designed and implemented SVG-based annotation tools.</li>
        <li>Mentored junior engineers and presented multiple talks about advanced concepts in JavaScript.</li>
        <li>Hired as Senior Software Engineer. Promoted to Lead Software Engineer in March 2018.</li>
      </Ul>
    </main>
  </Layout>
);

About.propTypes = {
  data: PropTypes.shape({
    site: siteShape,
  }).isRequired,
};

export default About;

export const aboutPageQuery = graphql`
  query AboutPageSiteMetadata {
    site {
      siteMetadata {
        title
      }
    }
  }
`;
